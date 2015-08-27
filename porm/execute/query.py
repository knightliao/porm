#!/usr/bin/env python
# coding=utf8


class QueryResultWrapper(object):
    """
    Provides an iterator over the results of a raw Query, additionally doing
    two things:
    - converts rows from the database into model instances
    - ensures that multiple iterations do not result in multiple queries
    """

    def __init__(self, model, cursor, meta=None, chunk_size=100):
        self.model = model
        self.cursor = cursor
        self.query_meta = meta or {}
        self.column_meta = self.query_meta.get('columns')
        self.join_meta = self.query_meta.get('graph')
        self.chunk_size = chunk_size

        # a query will be considered "simple" if it pulls columns straight
        # from the primary model being queried
        self.simple = self.query_meta.get('simple') or not self.column_meta

        if self.simple:
            cols = []
            non_cols = []
            for i in range(len(self.cursor.description)):
                col = self.cursor.description[i][0]
                if col in model._meta.columns:
                    cols.append((i, model._meta.columns[col]))
                else:
                    non_cols.append((i, col))
            self._cols = cols
            self._non_cols = non_cols
            self._iter_fn = self.simple_iter
        else:
            self._iter_fn = self.construct_instance

        self.__ct = 0
        self.__idx = 0

        self._result_cache = []
        self._populated = False

        self.__read_cache = []
        self.__read_idx = 0
        self.__read_ct = 0

    def simple_iter(self, row):
        instance = self.model()
        for i, f in self._cols:
            setattr(instance, f.name, f.python_value(row[i]))
        for i, f in self._non_cols:
            setattr(instance, f, row[i])
        return instance

    def construct_instance(self, row):
        # we have columns, models, and a graph of joins to reconstruct
        collected_models = {}
        for i, (model, col) in enumerate(self.column_meta):
            value = row[i]

            if isinstance(col, tuple):
                if len(col) == 3:
                    model = self.model  # special-case aggregates
                    col_name = attr = col[2]
                else:
                    col_name, attr = col
            else:
                col_name = attr = col

            if model not in collected_models:
                collected_models[model] = model()

            instance = collected_models[model]

            if col_name in instance._meta.columns:
                field = instance._meta.columns[col_name]
                setattr(instance, field.name, field.python_value(value))
            else:
                setattr(instance, attr, value)

        return self.follow_joins(self.join_meta, collected_models, self.model)

    def follow_joins(self, joins, collected_models, current):
        inst = collected_models[current]

        if current not in joins:
            return inst

        for joined_model, _, _ in joins[current]:
            if joined_model in collected_models:
                joined_inst = self.follow_joins(joins, collected_models, joined_model)
                fk_field = current._meta.get_related_field_for_model(joined_model)

                if not fk_field:
                    continue

                if not joined_inst.get_pk():
                    joined_inst.set_pk(getattr(inst, fk_field.id_storage))

                setattr(inst, fk_field.name, joined_inst)
                setattr(inst, fk_field.id_storage, joined_inst.get_pk())

        return inst

    def __iter__(self):
        self.__idx = self.__read_idx = 0

        if not self._populated:
            return self
        else:
            return iter(self._result_cache)

    def iterate(self):
        if self.__read_idx >= self.__read_ct:
            rows = self.cursor.fetchmany(self.chunk_size)
            self.__read_ct = len(rows)
            if self.__read_ct:
                self.__read_cache = rows
                self.__read_idx = 0
            else:
                self._populated = True
                raise StopIteration

        instance = self._iter_fn(self.__read_cache[self.__read_idx])
        self.__read_idx += 1
        return instance

    def iterator(self):
        while 1:
            yield self.iterate()

    def next(self):
        # check to see if we have a row in our instance cache
        if self.__idx < self.__ct:
            inst = self._result_cache[self.__idx]
            self.__idx += 1
            return inst

        instance = self.iterate()
        instance.prepared()  # <-- model prepared hook
        self._result_cache.append(instance)
        self.__ct += 1
        self.__idx += 1
        return instance


# create
class DoesNotExist(Exception):
    pass


# semantic wrappers for ordering the results of a `SelectQuery`
def asc(f):
    return (f, 'ASC')


def desc(f):
    return (f, 'DESC')


# wrappers for performing aggregation in a `SelectQuery`
def Count(f, alias='count'):
    return ('COUNT', f, alias)


def Max(f, alias='max'):
    return ('MAX', f, alias)


def Min(f, alias='min'):
    return ('MIN', f, alias)


def Sum(f, alias='sum'):
    return ('SUM', f, alias)


def Avg(f, alias='avg'):
    return ('AVG', f, alias)


# decorator for query methods to indicate that they change the state of the
# underlying data structures
def returns_clone(func):
    def inner(self, *args, **kwargs):
        clone = self.clone()
        res = func(clone, *args, **kwargs)
        return clone

    return inner

# helpers
ternary = lambda cond, t, f: (cond and [t] or [f])[0]


class Node(object):
    def __init__(self, connector='AND', children=None):
        self.connector = connector
        self.children = children or []
        self.negated = False

    def connect(self, rhs, connector):
        if isinstance(rhs, Leaf):
            if connector == self.connector:
                self.children.append(rhs)
                return self
            else:
                p = Node(connector)
                p.children = [self, rhs]
                return p
        elif isinstance(rhs, Node):
            p = Node(connector)
            p.children = [self, rhs]
            return p

    def __or__(self, rhs):
        return self.connect(rhs, 'OR')

    def __and__(self, rhs):
        return self.connect(rhs, 'AND')

    def __invert__(self):
        self.negated = not self.negated
        return self

    def __nonzero__(self):
        return bool(self.children)

    def __unicode__(self):
        query = []
        nodes = []
        for child in self.children:
            if isinstance(child, Q):
                query.append(unicode(child))
            elif isinstance(child, Node):
                nodes.append('(%s)' % unicode(child))
        query.extend(nodes)
        connector = ' %s ' % self.connector
        query = connector.join(query)
        if self.negated:
            query = 'NOT %s' % query
        return query


class Leaf(object):
    def __init__(self):
        self.parent = None

    def connect(self, connector):
        if self.parent is None:
            self.parent = Node(connector)
            self.parent.children.append(self)

    def __or__(self, rhs):
        self.connect('OR')
        return self.parent | rhs

    def __and__(self, rhs):
        self.connect('AND')
        return self.parent & rhs

    def __invert__(self):
        self.negated = not self.negated
        return self


class Q(Leaf):
    def __init__(self, _model=None, **kwargs):
        self.model = _model
        self.query = kwargs
        self.negated = False
        super(Q, self).__init__()

    def __unicode__(self):
        bits = ['%s = %s' % (k, v) for k, v in self.query.items()]
        if len(self.query.items()) > 1:
            connector = ' AND '
            expr = '(%s)' % connector.join(bits)
        else:
            expr = bits[0]
        if self.negated:
            expr = 'NOT %s' % expr
        return expr


class F(object):
    def __init__(self, field, model=None):
        self.field = field
        self.model = model
        self.op = None
        self.value = None

    def __add__(self, rhs):
        self.op = '+'
        self.value = rhs
        return self

    def __sub__(self, rhs):
        self.op = '-'
        self.value = rhs
        return self


class R(Leaf):
    def __init__(self, *params):
        self.params = params
        super(R, self).__init__()

    def sql_select(self, model_class):
        if len(self.params) == 2:
            return self.params
        else:
            raise ValueError('Incorrect number of argument provided for R() expression')

    def sql_where(self):
        return self.params[0], self.params[1:]

    def sql_update(self):
        return self.params[0], self.params[1]


def apply_model(model, item):
    """
    Q() objects take a model, which provides context for the keyword arguments.
    In this way Q() objects can be mixed across models.  The purpose of this
    function is to recurse into a query datastructure and apply the given model
    to all Q() objects that do not have a model explicitly set.
    """
    if isinstance(item, Node):
        for child in item.children:
            apply_model(model, child)
    elif isinstance(item, Q):
        if item.model is None:
            item.model = model


def parseq(model, *args, **kwargs):
    """
    Convert any query into a single Node() object -- used to build up the list
    of where clauses when querying.
    """
    node = Node()

    for piece in args:
        apply_model(model, piece)
        if isinstance(piece, (Q, R, Node)):
            node.children.append(piece)
        else:
            raise TypeError('Unknown object: %s' % piece)

    if kwargs:
        node.children.append(Q(model, **kwargs))

    return node


def find_models(item):
    """
    Utility function to find models referenced in a query and return a set()
    containing them.  This function is used to generate the list of models that
    are part of a where clause.
    """
    seen = set()
    if isinstance(item, Node):
        for child in item.children:
            seen.update(find_models(child))
    elif isinstance(item, Q):
        seen.add(item.model)
    return seen


class EmptyResultException(Exception):
    pass


class BaseQuery(object):
    query_separator = '__'
    force_alias = False
    require_commit = True

    def __init__(self, model):
        self.model = model
        self.query_context = model
        self.database = self.model._meta.database
        self.operations = self.database.adapter.operations
        self.interpolation = self.database.adapter.interpolation

        self._dirty = True
        self._where = []
        self._where_models = set()
        self._joins = {}
        self._joined_models = set()
        self._table_alias = {}

    def _clone_dict_graph(self, dg):
        cloned = {}
        for node, edges in dg.items():
            cloned[node] = list(edges)
        return cloned

    def clone_where(self):
        return list(self._where)

    def clone_joins(self):
        return self._clone_dict_graph(self._joins)

    def clone(self):
        raise NotImplementedError

    def qn(self, name):
        return self.database.quote_name(name)

    def lookup_cast(self, field, lookup, value):
        return self.database.adapter.lookup_cast(field, lookup, value)

    def parse_query_args(self, _model, **query):
        """
        Parse out and normalize clauses in a query.  The query is composed of
        various column+lookup-type/value pairs.  Validates that the lookups
        are valid and returns a list of lookup tuples that have the form:
        (field name, (operation, value))
        """
        model = _model
        parsed = []
        for lhs, rhs in query.iteritems():
            if self.query_separator in lhs:
                lhs, op = lhs.rsplit(self.query_separator, 1)
            else:
                op = 'eq'

            if lhs in model._meta.columns:
                lhs = model._meta.columns[lhs].name

            try:
                field = model._meta.get_field_by_name(lhs)
            except AttributeError:
                field = model._meta.get_related_field_by_name(lhs)
                if field is None:
                    raise

            op = self.database.adapter.op_override(field, op, rhs)

            if isinstance(rhs, R):
                expr, params = rhs.sql_where()
                lookup_value = [field.db_value(o) for o in params]

                combined_expr = self.operations[op] % expr
                operation = combined_expr % tuple(self.interpolation for p in params)
            elif isinstance(rhs, F):
                lookup_value = rhs
                operation = self.operations[op]  # leave as "%s"
            else:
                if op == 'in':
                    if isinstance(rhs, SelectQuery):
                        lookup_value = rhs
                        operation = 'IN (%s)'
                    else:
                        if not rhs:
                            raise EmptyResultException
                        lookup_value = [field.db_value(o) for o in rhs]
                        operation = self.operations[op] % \
                                    (','.join([self.interpolation for v in lookup_value]))
                elif op == 'is':
                    if rhs is not None:
                        raise ValueError('__is lookups only accept None')
                    operation = 'IS NULL'
                    lookup_value = []
                elif op == 'isnull':
                    operation = 'IS NULL' if rhs else 'IS NOT NULL'
                    lookup_value = []
                elif op == 'between':
                    lookup_value = [field.db_value(o) for o in rhs]
                    operation = self.operations[op] % (self.interpolation, self.interpolation)
                elif isinstance(rhs, (list, tuple)):
                    lookup_value = [field.db_value(o) for o in rhs]
                    operation = self.operations[op] % self.interpolation
                else:
                    lookup_value = field.db_value(rhs)
                    operation = self.operations[op] % self.interpolation

            parsed.append(
                (field.db_column, (operation, self.lookup_cast(field, op, lookup_value)))
            )

        return parsed

    @returns_clone
    def where(self, *args, **kwargs):
        parsed = parseq(self.query_context, *args, **kwargs)
        if parsed:
            self._where.append(parsed)
            self._where_models.update(find_models(parsed))

    @returns_clone
    def join(self, model, join_type=None, on=None, alias=None):
        if self.query_context._meta.rel_exists(model):
            self._joined_models.add(model)
            self._joins.setdefault(self.query_context, [])
            self._joins[self.query_context].append((model, join_type, on))
            if alias:
                self._table_alias[model] = alias
            self.query_context = model
        else:
            raise AttributeError('No foreign key found between %s and %s' % \
                                 (self.query_context.__name__, model.__name__))

    @returns_clone
    def switch(self, model):
        if model == self.model:
            self.query_context = model
            return

        if model in self._joined_models:
            self.query_context = model
            return
        raise AttributeError('You must JOIN on %s' % model.__name__)

    def use_aliases(self):
        return len(self._joined_models) > 0 or self.force_alias

    def combine_field(self, alias, field_col):
        quoted = self.qn(field_col)
        if alias:
            return '%s.%s' % (alias, quoted)
        return quoted

    def safe_combine(self, model, alias, col):
        if col in model._meta.columns:
            return self.combine_field(alias, col)
        elif col in model._meta.fields:
            return self.combine_field(alias, model._meta.fields[col].db_column)
        return col

    def follow_joins(self, current, alias_map, alias_required, alias_count, seen=None):
        computed = []
        seen = seen or set()

        if current not in self._joins:
            return computed, alias_count

        for i, (model, join_type, on) in enumerate(self._joins[current]):
            seen.add(model)

            if alias_required:
                if model in self._table_alias:
                    alias_map[model] = self._table_alias[model]
                else:
                    alias_count += 1
                    alias_map[model] = 't%d' % alias_count
            else:
                alias_map[model] = ''

            from_model = current
            field = from_model._meta.get_related_field_for_model(model, on)
            if field:
                left_field = field.db_column
                right_field = model._meta.pk_col
            else:
                field = from_model._meta.get_reverse_related_field_for_model(model, on)
                left_field = from_model._meta.pk_col
                right_field = field.db_column

            if join_type is None:
                if field.null and model not in self._where_models:
                    join_type = 'LEFT OUTER'
                else:
                    join_type = 'INNER'

            computed.append(
                '%s JOIN %s AS %s ON %s = %s' % (
                    join_type,
                    self.qn(model._meta.db_table),
                    alias_map[model],
                    self.combine_field(alias_map[from_model], left_field),
                    self.combine_field(alias_map[model], right_field),
                )
            )

            joins, alias_count = self.follow_joins(model, alias_map, alias_required, alias_count, seen)
            computed.extend(joins)

        return computed, alias_count

    def compile_where(self):
        alias_count = 0
        alias_map = {}

        alias_required = self.use_aliases()
        if alias_required:
            if self.model in self._table_alias:
                alias_map[self.model] = self._table_alias[self.model]
            else:
                alias_count += 1
                alias_map[self.model] = 't%d' % alias_count
        else:
            alias_map[self.model] = ''

        computed_joins, _ = self.follow_joins(self.model, alias_map, alias_required, alias_count)

        clauses = [self.parse_node(node, alias_map) for node in self._where]

        return computed_joins, clauses, alias_map

    def flatten_clauses(self, clauses):
        where_with_alias = []
        where_data = []
        for query, data in clauses:
            where_with_alias.append(query)
            where_data.extend(data)
        return where_with_alias, where_data

    def convert_where_to_params(self, where_data):
        flattened = []
        for clause in where_data:
            if isinstance(clause, (tuple, list)):
                flattened.extend(clause)
            else:
                flattened.append(clause)
        return flattened

    def parse_node(self, node, alias_map):
        query = []
        query_data = []
        for child in node.children:
            if isinstance(child, Q):
                parsed, data = self.parse_q(child, alias_map)
                query.append(parsed)
                query_data.extend(data)
            elif isinstance(child, R):
                parsed, data = self.parse_r(child, alias_map)
                query.append(parsed % tuple(self.interpolation for o in data))
                query_data.extend(data)
            elif isinstance(child, Node):
                parsed, data = self.parse_node(child, alias_map)
                query.append('(%s)' % parsed)
                query_data.extend(data)
        connector = ' %s ' % node.connector
        query = connector.join(query)
        if node.negated:
            query = 'NOT (%s)' % query
        return query, query_data

    def parse_q(self, q, alias_map):
        model = q.model or self.model
        query = []
        query_data = []
        parsed = self.parse_query_args(model, **q.query)
        for (name, lookup) in parsed:
            operation, value = lookup
            if isinstance(value, SelectQuery):
                sql, value = self.convert_subquery(value)
                operation = operation % sql

            if isinstance(value, F):
                f_model = value.model or model
                operation = operation % self.parse_f(value, f_model, alias_map)
            else:
                query_data.append(value)

            combined = self.combine_field(alias_map[model], name)
            query.append('%s %s' % (combined, operation))

        if len(query) > 1:
            query = '(%s)' % (' AND '.join(query))
        else:
            query = query[0]

        if q.negated:
            query = 'NOT %s' % query

        return query, query_data

    def parse_f(self, f_object, model, alias_map):
        combined = self.combine_field(alias_map[model], f_object.field)
        if f_object.op is not None:
            combined = '(%s %s %s)' % (combined, f_object.op, f_object.value)

        return combined

    def parse_r(self, r_object, alias_map):
        return r_object.sql_where()

    def convert_subquery(self, subquery):
        orig_query = subquery.query
        if subquery.query == '*':
            subquery.query = subquery.model._meta.pk_name

        subquery.force_alias, orig_alias = True, subquery.force_alias
        sql, data = subquery.sql()
        subquery.query = orig_query
        subquery.force_alias = orig_alias
        return sql, data

    def sorted_models(self, alias_map):
        return [
            (model, alias) \
            for (model, alias) in sorted(alias_map.items(), key=lambda i: i[1])
        ]

    def sql(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError

    def raw_execute(self, query, params):
        return self.database.execute(query, params, self.require_commit)


class RawQuery(BaseQuery):
    def __init__(self, model, query, *params):
        self._sql = query
        self._params = list(params)
        super(RawQuery, self).__init__(model)

    def clone(self):
        return RawQuery(self.model, self._sql, *self._params)

    def sql(self):
        return self._sql, self._params

    def execute(self):
        return QueryResultWrapper(self.model, self.raw_execute(*self.sql()))

    def join(self):
        raise AttributeError('Raw queries do not support joining programmatically')

    def where(self):
        raise AttributeError('Raw queries do not support querying programmatically')

    def switch(self):
        raise AttributeError('Raw queries do not support switching contexts')

    def __iter__(self):
        return iter(self.execute())


class SelectQuery(BaseQuery):
    require_commit = False

    def __init__(self, model, query=None):
        self.query = query or '*'
        self._group_by = []
        self._having = []
        self._order_by = []
        self._limit = None
        self._offset = None
        self._distinct = False
        self._qr = None
        self._for_update = False
        self._naive = False
        super(SelectQuery, self).__init__(model)

    def clone(self):
        query = SelectQuery(self.model, self.query)
        query.query_context = self.query_context
        query._group_by = list(self._group_by)
        query._having = list(self._having)
        query._order_by = list(self._order_by)
        query._limit = self._limit
        query._offset = self._offset
        query._distinct = self._distinct
        query._qr = self._qr
        query._for_update = self._for_update
        query._naive = self._naive
        query._where = self.clone_where()
        query._where_models = set(self._where_models)
        query._joined_models = self._joined_models.copy()
        query._joins = self.clone_joins()
        query._table_alias = dict(self._table_alias)
        return query

    @returns_clone
    def paginate(self, page, paginate_by=20):
        if page > 0:
            page -= 1
        self._limit = paginate_by
        self._offset = page * paginate_by

    @returns_clone
    def limit(self, num_rows):
        self._limit = num_rows

    @returns_clone
    def offset(self, num_rows):
        self._offset = num_rows

    @returns_clone
    def for_update(self, for_update=True):
        self._for_update = for_update

    def count(self):
        if self._distinct or self._group_by:
            return self.wrapped_count()

        clone = self.order_by()
        clone._limit = clone._offset = None

        if clone.use_aliases():
            clone.query = 'COUNT(t1.%s)' % (clone.model._meta.pk_col)
        else:
            clone.query = 'COUNT(%s)' % (clone.model._meta.pk_col)

        res = clone.database.execute(*clone.sql(), require_commit=False)

        return (res.fetchone() or [0])[0]

    def wrapped_count(self):
        clone = self.order_by()
        clone._limit = clone._offset = None

        sql, params = clone.sql()
        query = 'SELECT COUNT(1) FROM (%s) AS wrapped_select' % sql

        res = clone.database.execute(query, params, require_commit=False)

        return res.fetchone()[0]

    @returns_clone
    def group_by(self, *clauses):
        model = self.query_context
        from porm.db.models import Model

        for clause in clauses:
            if isinstance(clause, basestring):
                fields = (clause,)
            elif isinstance(clause, (list, tuple)):
                fields = clause
            elif issubclass(clause, Model):
                model = clause
                fields = clause._meta.get_field_names()

            self._group_by.append((model, fields))

    @returns_clone
    def having(self, *clauses):
        self._having = clauses

    @returns_clone
    def distinct(self):
        self._distinct = True

    @returns_clone
    def order_by(self, *clauses):
        from porm.db.fields import Field

        order_by = []

        for clause in clauses:
            if isinstance(clause, tuple):
                if len(clause) == 3:
                    model, field, ordering = clause
                elif len(clause) == 2:
                    if isinstance(clause[0], basestring):
                        model = self.query_context
                        field, ordering = clause
                    else:
                        model, field = clause
                        ordering = 'ASC'
                else:
                    raise ValueError('Incorrect arguments passed in order_by clause')
            elif isinstance(clause, basestring):
                model = self.query_context
                field = clause
                ordering = 'ASC'
            elif isinstance(clause, Field):
                model = clause.model
                field = clause.name
                ordering = 'ASC'
            else:
                raise ValueError('Unknown value passed in to order_by')

            order_by.append(
                (model, field, ordering)
            )

        self._order_by = order_by

    def exists(self):
        clone = self.paginate(1, 1)
        clone.query = '(1) AS a'
        curs = self.database.execute(*clone.sql(), require_commit=False)
        return bool(curs.fetchone())

    def get(self, *args, **kwargs):
        orig_ctx = self.query_context
        self.query_context = self.model
        query = self.where(*args, **kwargs).paginate(1, 1)
        try:
            obj = query.execute().next()
            return obj
        except StopIteration:
            raise self.model.DoesNotExist('instance matching query does not exist:\nSQL: %s\nPARAMS: %s' % (
                query.sql()
            ))
        finally:
            self.query_context = orig_ctx

    def filter(self, *args, **kwargs):
        return filter_query(self, *args, **kwargs)

    def annotate(self, related_model, aggregation=None):
        return annotate_query(self, related_model, aggregation)

    def aggregate(self, func):
        clone = self.order_by()
        clone.query = [func]
        curs = self.database.execute(*clone.sql(), require_commit=False)
        return curs.fetchone()[0]

    @returns_clone
    def naive(self, make_naive=True):
        self._naive = make_naive

    def parse_select_query(self, alias_map):
        q = self.query
        models_queried = 0
        local_columns = True

        if isinstance(q, (list, tuple)):
            q = {self.model: self.query}
        elif isinstance(q, basestring):
            # convert '*' and primary key lookups
            if q == '*':
                q = {self.model: self.model._meta.get_field_names()}
            elif q in (self.model._meta.pk_col, self.model._meta.pk_name):
                q = {self.model: [self.model._meta.pk_name]}
            else:
                return q, [], [], False

        # by now we should have a dictionary if a valid type was passed in
        if not isinstance(q, dict):
            raise TypeError('Unknown type encountered parsing select query')

        # gather aliases and models
        sorted_models = self.sorted_models(alias_map)

        # normalize if we are working with a dictionary
        columns = []
        model_cols = []
        sparams = []

        for model, alias in sorted_models:
            if model not in q:
                continue

            models_queried += 1

            if '*' in q[model]:
                idx = q[model].index('*')
                q[model] = q[model][:idx] + model._meta.get_field_names() + q[model][idx + 1:]

            for clause in q[model]:
                if hasattr(clause, 'sql_select'):
                    clause = clause.sql_select(model)

                if isinstance(clause, tuple):
                    local_columns = False
                    if len(clause) > 3:
                        template, col_name, col_alias = clause[:3]
                        cparams = clause[3:]
                        column = model._meta.get_column(col_name)
                        columns.append(template % \
                                       (self.safe_combine(model, alias, column), col_alias)
                        )
                        sparams.extend(cparams)
                        model_cols.append((model, (template, column, col_alias)))
                    elif len(clause) == 3:
                        func, col_name, col_alias = clause
                        column = model._meta.get_column(col_name)
                        columns.append('%s(%s) AS %s' % \
                                       (func, self.safe_combine(model, alias, column), col_alias)
                        )
                        model_cols.append((model, (func, column, col_alias)))
                    elif len(clause) == 2:
                        col_name, col_alias = clause
                        column = model._meta.get_column(col_name)
                        columns.append('%s AS %s' % \
                                       (self.safe_combine(model, alias, column), col_alias)
                        )
                        model_cols.append((model, (column, col_alias)))
                    else:
                        raise ValueError('Unknown type in select query')
                else:
                    column = model._meta.get_column(clause)
                    columns.append(self.safe_combine(model, alias, column))
                    model_cols.append((model, column))

        return ', '.join(columns), model_cols, sparams, (models_queried == 1 and local_columns)

    def sql_meta(self):
        joins, clauses, alias_map = self.compile_where()
        where, where_data = self.flatten_clauses(clauses)

        table = self.qn(self.model._meta.db_table)

        params = []
        group_by = []
        use_aliases = self.use_aliases()

        if use_aliases:
            table = '%s AS %s' % (table, alias_map[self.model])

        for model, clause in self._group_by:
            if use_aliases:
                alias = alias_map[model]
            else:
                alias = ''

            for field in clause:
                group_by.append(self.safe_combine(model, alias, field))

        parsed_query, model_cols, sparams, simple = self.parse_select_query(alias_map)
        params.extend(sparams)
        query_meta = {
            'columns': model_cols,
            'graph': self._joins,
            'simple': simple,
        }

        if self._distinct:
            sel = 'SELECT DISTINCT'
        else:
            sel = 'SELECT'

        select = '%s %s FROM %s' % (sel, parsed_query, table)
        joins = '\n'.join(joins)
        where = ' AND '.join(where)
        group_by = ', '.join(group_by)
        having = ' AND '.join(self._having)

        order_by = []
        for piece in self._order_by:
            model, field, ordering = piece
            if use_aliases:
                alias = alias_map[model]
            else:
                alias = ''

            order_by.append('%s %s' % (self.safe_combine(model, alias, field), ordering))

        pieces = [select]

        if joins:
            pieces.append(joins)
        if where:
            pieces.append('WHERE %s' % where)
            params.extend(self.convert_where_to_params(where_data))

        if group_by:
            pieces.append('GROUP BY %s' % group_by)
        if having:
            pieces.append('HAVING %s' % having)
        if order_by:
            pieces.append('ORDER BY %s' % ', '.join(order_by))
        if self._limit:
            pieces.append('LIMIT %d' % self._limit)
        if self._offset:
            pieces.append('OFFSET %d' % self._offset)

        if self._for_update and self.database.adapter.for_update_support:
            pieces.append('FOR UPDATE')

        return ' '.join(pieces), params, query_meta

    def sql(self):
        query, params, meta = self.sql_meta()
        return query, params

    def execute(self):
        if self._dirty or not self._qr:
            try:
                sql, params, meta = self.sql_meta()
            except EmptyResultException:
                return []
            else:
                if self._naive:
                    meta = None
                self._qr = QueryResultWrapper(self.model, self.raw_execute(sql, params), meta)
                self._dirty = False
                return self._qr
        else:
            # call the __iter__ method directly
            return self._qr

    def __iter__(self):
        return iter(self.execute())


class UpdateQuery(BaseQuery):
    def __init__(self, _model, **kwargs):
        self.update_query = kwargs
        super(UpdateQuery, self).__init__(_model)

    def clone(self):
        query = UpdateQuery(self.model, **self.update_query)
        query._where = self.clone_where()
        query._where_models = set(self._where_models)
        query._joined_models = self._joined_models.copy()
        query._joins = self.clone_joins()
        query._table_alias = dict(self._table_alias)
        return query

    def parse_update(self):
        sets = {}
        for k, v in self.update_query.iteritems():
            if k in self.model._meta.columns:
                k = self.model._meta.columns[k].name

            try:
                field = self.model._meta.get_field_by_name(k)
            except AttributeError:
                field = self.model._meta.get_related_field_by_name(k)
                if field is None:
                    raise

            if not isinstance(v, (F, R)):
                v = field.db_value(v)

            sets[field.db_column] = v

        return sets

    def sql(self):
        joins, clauses, alias_map = self.compile_where()
        where, where_data = self.flatten_clauses(clauses)
        set_statement = self.parse_update()

        params = []
        update_params = []

        alias = alias_map.get(self.model)

        for k, v in sorted(set_statement.items(), key=lambda (k, v): k):
            if isinstance(v, F):
                value = self.parse_f(v, v.model or self.model, alias_map)
            elif isinstance(v, R):
                value, rparams = v.sql_update()
                value = value % self.interpolation
                params.append(rparams)
            else:
                params.append(v)
                value = self.interpolation

            update_params.append('%s=%s' % (self.combine_field(alias, k), value))

        update = 'UPDATE %s SET %s' % (
            self.qn(self.model._meta.db_table), ', '.join(update_params))
        where = ' AND '.join(where)

        pieces = [update]

        if where:
            pieces.append('WHERE %s' % where)
            params.extend(self.convert_where_to_params(where_data))

        return ' '.join(pieces), params

    def join(self, *args, **kwargs):
        raise AttributeError('Update queries do not support JOINs in sqlite')

    def execute(self):
        result = self.raw_execute(*self.sql())
        return self.database.rows_affected(result)


class DeleteQuery(BaseQuery):
    def clone(self):
        query = DeleteQuery(self.model)
        query._where = self.clone_where()
        query._where_models = set(self._where_models)
        query._joined_models = self._joined_models.copy()
        query._joins = self.clone_joins()
        query._table_alias = dict(self._table_alias)
        return query

    def sql(self):
        joins, clauses, alias_map = self.compile_where()
        where, where_data = self.flatten_clauses(clauses)

        params = []

        delete = 'DELETE FROM %s' % (self.qn(self.model._meta.db_table))
        where = ' AND '.join(where)

        pieces = [delete]

        if where:
            pieces.append('WHERE %s' % where)
            params.extend(self.convert_where_to_params(where_data))

        return ' '.join(pieces), params

    def join(self, *args, **kwargs):
        raise AttributeError('Update queries do not support JOINs in sqlite')

    def execute(self):
        result = self.raw_execute(*self.sql())
        return self.database.rows_affected(result)


class InsertQuery(BaseQuery):
    def __init__(self, _model, **kwargs):
        self.insert_query = kwargs
        super(InsertQuery, self).__init__(_model)

    def parse_insert(self):
        cols = []
        vals = []
        for k, v in sorted(self.insert_query.items(), key=lambda (k, v): k):
            if k in self.model._meta.columns:
                k = self.model._meta.columns[k].name

            try:
                field = self.model._meta.get_field_by_name(k)
            except AttributeError:
                field = self.model._meta.get_related_field_by_name(k)
                if field is None:
                    raise

            cols.append(self.qn(field.db_column))
            vals.append(field.db_value(v))

        return cols, vals

    def sql(self):
        cols, vals = self.parse_insert()

        insert = 'INSERT INTO %s (%s) VALUES (%s)' % (
            self.qn(self.model._meta.db_table),
            ','.join(cols),
            ','.join(self.interpolation for v in vals)
        )

        return insert, vals

    def where(self, *args, **kwargs):
        raise AttributeError('Insert queries do not support WHERE clauses')

    def join(self, *args, **kwargs):
        raise AttributeError('Insert queries do not support JOINs')

    def execute(self):
        result = self.raw_execute(*self.sql())
        return self.database.last_insert_id(result, self.model)


def model_or_select(m_or_q):
    """
    Return both a model and a select query for the provided model *OR* select
    query.
    """
    if isinstance(m_or_q, BaseQuery):
        return (m_or_q.model, m_or_q)
    else:
        return (m_or_q, m_or_q.select())


def convert_lookup(model, joins, lookup):
    """
    Given a model, a graph of joins, and a lookup, return a tuple containing
    a normalized lookup:

    (model actually being queried, updated graph of joins, normalized lookup)
    """
    operations = model._meta.database.adapter.operations

    pieces = lookup.split('__')
    operation = None

    query_model = model

    if len(pieces) > 1:
        if pieces[-1] in operations:
            operation = pieces.pop()

        lookup = pieces.pop()

        # we have some joins
        if len(pieces):
            for piece in pieces:
                # piece is something like 'blog' or 'entry_set'
                joined_model = None
                for field in query_model._meta.get_fields():
                    from porm.db.fields import Field, ForeignKeyField

                    if not isinstance(field, ForeignKeyField):
                        continue

                    if piece in (field.name, field.db_column, field.related_name):
                        joined_model = field.to

                if not joined_model:
                    try:
                        joined_model = query_model._meta.reverse_relations[piece]
                    except KeyError:
                        raise ValueError('Unknown relation: "%s" of "%s"' % (
                            piece,
                            query_model,
                        ))

                joins.setdefault(query_model, set())
                joins[query_model].add(joined_model)
                query_model = joined_model

    if operation:
        lookup = '%s__%s' % (lookup, operation)

    return query_model, joins, lookup


def filter_query(model_or_query, *args, **kwargs):
    """
    Provide a django-like interface for executing queries
    """
    model, select_query = model_or_select(model_or_query)

    query = {}  # mapping of models to queries
    joins = {}  # a graph of joins needed, passed into the convert_lookup function

    # traverse Q() objects, find any joins that may be lurking -- clean up the
    # lookups and assign the correct model
    def fix_q(node_or_q, joins):
        if isinstance(node_or_q, Node):
            for child in node_or_q.children:
                fix_q(child, joins)
        elif isinstance(node_or_q, Q):
            new_query = {}
            curr_model = node_or_q.model or model
            for raw_lookup, value in node_or_q.query.items():
                query_model, joins, lookup = convert_lookup(curr_model, joins, raw_lookup)
                new_query[lookup] = value
            node_or_q.model = query_model
            node_or_q.query = new_query

    for node_or_q in args:
        fix_q(node_or_q, joins)

    # iterate over keyword lookups and determine lookups and necessary joins
    for raw_lookup, value in kwargs.items():
        queried_model, joins, lookup = convert_lookup(model, joins, raw_lookup)
        query.setdefault(queried_model, [])
        query[queried_model].append((lookup, value))

    def follow_joins(current, query):
        if current in joins:
            for joined_model in joins[current]:
                query = query.switch(current)
                if joined_model not in query._joined_models:
                    query = query.join(joined_model)
                query = follow_joins(joined_model, query)
        return query

    select_query = follow_joins(model, select_query)

    for node in args:
        select_query = select_query.where(node)

    for model, lookups in query.items():
        qargs, qkwargs = [], {}
        for lookup in lookups:
            if isinstance(lookup, tuple):
                qkwargs[lookup[0]] = lookup[1]
            else:
                qargs.append(lookup)
        select_query = select_query.switch(model).where(*qargs, **qkwargs)

    return select_query


def annotate_query(select_query, related_model, aggregation):
    """
    Perform an aggregation against a related model
    """
    aggregation = aggregation or Count(related_model._meta.pk_name)
    model = select_query.model

    select_query = select_query.switch(model)
    cols = select_query.query

    # ensure the join is there
    if related_model not in select_query._joined_models:
        select_query = select_query.join(related_model).switch(model)

    # query for it
    if isinstance(cols, dict):
        selection = cols
        group_by = cols[model]
    elif isinstance(cols, basestring):
        selection = {model: [cols]}
        if cols == '*':
            group_by = model
        else:
            group_by = [col.strip() for col in cols.split(',')]
    elif isinstance(cols, (list, tuple)):
        selection = {model: cols}
        group_by = cols
    else:
        raise ValueError('Unknown type passed in to select query: "%s"' % type(cols))

    # query for the related object
    if related_model in selection:
        selection[related_model].append(aggregation)
    else:
        selection[related_model] = [aggregation]

    select_query.query = selection
    if group_by == ['*']:
        return select_query
    else:
        return select_query.group_by(group_by)
