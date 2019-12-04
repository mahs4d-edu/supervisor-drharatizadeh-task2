class MahdiMatrix:
    def __init__(self, dimen_x, dimen_y, scalar_p=1, default_value=0):
        self.dimen_x = dimen_x
        self.dimen_y = dimen_y
        self._rows = {}
        self._columns = {}
        self.scalar_p = scalar_p
        self.default_value = default_value

    def set_item(self, x, y, value):
        t = False

        if x not in self._rows:
            self._rows[x] = {}
            t = True

        if y not in self._columns:
            self._columns[y] = {}
            t = True

        self._rows[x][y] = value
        self._columns[y][x] = value

        return t

    def get_row(self, x):
        if x in self._rows:
            return self._rows[x]
        else:
            return None

    def get_column(self, y):
        if y in self._columns:
            return self._columns[y]
        else:
            return None

    def get_item(self, x, y):
        if x in self._rows and y in self._rows[x]:
            return self._rows[x][y] * self.scalar_p

        return self.default_value * self.scalar_p

    def get_size(self):
        return self.dimen_x, self.dimen_y

    def multiply_scalar(self, p):
        self.scalar_p *= p

    def multiply(self, m2):
        if self.dimen_y != m2.dimen_x:
            raise ValueError('invalid multiplication')

        ret = MahdiMatrix(self.dimen_x, m2.dimen_y)

        for i in self._rows:
            for j in m2._columns:
                sum = 0
                for k in self._rows[i]:
                    sum += self.get_item(i, k) * m2.get_item(k, j)

                if sum != 0:
                    ret.set_item(i, j, sum)

        return ret

    def add_item(self, x, y, value):
        if x not in self._rows or y not in self._rows[x]:
            self.set_item(x, y, value)
        else:
            self._rows[x][y] += value
            self._columns[y][x] += value

    def __str__(self):
        s = ''
        for row in self._rows:
            for column in self._rows[row]:
                s += '({0}, {1}) = {2}\n'.format(row, column, self.get_item(row, column))

        return s


class MahdiGraph:
    def __init__(self, nodes_count):
        self.nodes_count = nodes_count
        self.adj_matrix = MahdiMatrix(nodes_count, nodes_count)
        self.nodes_k = {}

    def add_edge(self, a, b):
        self.adj_matrix.set_item(a, b, 1)
        self.adj_matrix.set_item(b, a, 1)

        if a not in self.nodes_k:
            self.nodes_k[a] = 1
        else:
            self.nodes_k[a] += 1

        if b not in self.nodes_k:
            self.nodes_k[b] = 1
        else:
            self.nodes_k[b] += 1

    def generate_t_matrix(self):
        self.t_matrix = MahdiMatrix(self.nodes_count, self.nodes_count)

        for row in self.adj_matrix._rows:
            for col in self.adj_matrix._rows[row]:
                self.t_matrix.add_item(col, row, self.adj_matrix.get_item(row, col) / self.nodes_k[row])

        del self.adj_matrix

    def pagerank(self, alpha, p_node):
        ppr = MahdiMatrix(self.nodes_count, 1, 1, 1 / self.nodes_count)
        self.t_matrix.multiply_scalar(alpha)
        for i in range(10):
            ppr = self.t_matrix.multiply(ppr)
            ppr.add_item(p_node, 0, 1 - alpha)

        return ppr
