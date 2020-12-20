class SmartVector(object):
    """
    Represents a smart 2d vector (x, y)
    """
    @staticmethod
    def from_coordinate(coordinate):
        return SmartVector(coordinate.x, coordinate.y)

    @staticmethod
    def get_north():
        return SmartVector(0, 1)

    @staticmethod
    def get_east():
        return SmartVector(1, 0)

    @staticmethod
    def get_south():
        return SmartVector(0, -1)

    @staticmethod
    def get_west():
        return SmartVector(-1, 0)

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __neg__(self):
        return SmartVector(-self.x, -self.y)

    def __add__(self, other):
        return SmartVector(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        return SmartVector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, int):
            return SmartVector(self.x * other, self.y * other)
        else:
            return self

    def __rmul__(self, other):
        if isinstance(other, int):
            return SmartVector(self.x * other, self.y * other)
        else:
            return self

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"SmartVector: (x = {self.x}, y = {self.y})"

    def __repr__(self):
        return f"SmartVector: (x = {self.x}, y = {self.y})"

    def get_perpendicular_vector_clockwise(self):
        """
        returns the corresponding perpendicular 2D vector (in a clockwise manner)
        """
        return SmartVector(self.y, -self.x)

    def get_perpendicular_vectors(self):
        """
        returns both corresponding perpendicular 2D vector
        """
        return [SmartVector(self.y, -self.x), SmartVector(-self.y, self.x)]


class SmartCoordinate(object):
    """
    Represents a smart cartesian coordinate (x, y)
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __neg__(self):
        return SmartCoordinate(-self.x, -self.y)

    def __add__(self, other):
        # works both adding coordinate + coordinate and also coordinate + vector 
        # (*because they have the same fields)
        return SmartCoordinate(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        if other == 0:
            return self
        elif isinstance(other, SmartVector):
            return self.__add__(SmartCoordinate(other.x, other.y))
        else:
            return self.__add__(other)

    def __sub__(self, other):
        # works both subtracting coordinate - coordinate and also coordinate - vector 
        # (*because they have the same fields)
        return SmartCoordinate(self.x - other.x, self.y - other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.y < other.y if not self.y == other.y else self.x < other.x

    def __str__(self):
        return f"SmartCoordinate: (x = {self.x}, y = {self.y})"

    def __repr__(self):
        return f"SmartCoordinate: (x = {self.x}, y = {self.y})"
