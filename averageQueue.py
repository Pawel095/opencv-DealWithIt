import statistics


class averageQueue:

    def __init__(self, maxSize=20):
        self.__maxSize = maxSize
        self.__queue = list()

    def add(self, a):
        """enqueues an item; if the items in the queue are equal or exceed the maxSize then the oldest is removed.
        if a None is passed one item is removed."""
        if len(self.__queue) < self.__maxSize:
            if a is not None:
                self.__queue.append(a)
            else:
                self.__queue.pop(0)
        else:
            self.__queue.append(a)
            self.__queue.pop(0)

    def get_avg(self):
        """:returns average of all values without removing any from the queue"""
        return statistics.mean(self.__queue)
