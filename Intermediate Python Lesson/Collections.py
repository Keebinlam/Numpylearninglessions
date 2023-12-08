# collections, counter, namedtuple, ordered dict, deque
# collection is a module that implements specialized container datatype providing alternative to dict, list, set, tuple.
# collection are useful for data processing, managing complex, data strcuture, improving the performance of the code
from collections import Counter
from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict
from collections import deque

# Counter: Stores elements as dictionary keys and their counter are stored as dictionary values

a = "bbbbaaaaaccccccc"
mycounter = Counter(a)
print(mycounter)
# Counter({'c': 7, 'a': 5, 'b': 4})
print(mycounter.items())
print(mycounter.keys())
print(mycounter.values())

# to show most common element in dictionary
a = "bbbbaaaaaccccccc"
mycounter = Counter(a)
print(mycounter.most_common(1))  # putting 1 gives us the most common

# to make the items in the counter a list
b = list(mycounter.elements())
print(b)

# Namedtuple: It generates a new subclass of tuple with named fields, considered lightweigtht
point = namedtuple('point', 'x,y')
pt = point(1, -4)
print(pt)

# Ordereddict: A subclass of dict that remembers the order in which its contents are added, even if a new entry overwrites an existing entry.
ordereddict = OrderedDict()
ordereddict['a'] = 1
ordereddict['b'] = 2
ordereddict['c'] = 3
ordereddict['d'] = 4
print(ordereddict)
# using python3, normal dictionary already has this feature of rememebering the order of the dictionary

# defaultdict: Similar to the regular dictionary (dict), but it provides a default value for the key that does not exist.
d = defaultdict(int)
d['a'] = 1
d['b'] = 2
print(d['a'])
# here when printing out the 'a' key, the value output will be, but now putting in a key that does not exsisit like 'c', the value will return with an int (0)
print(d['c'])

# deque: Short for "double-ended queue," it is designed to have fast appends and pops from both ends, as compared to list which is optimized for fast fixed-length operations and has slow append and pop operations as the length of the list increases.
f = deque()
f.append(1)
f.append(2)
f.appendleft(3)  # this will append elements on the left of the list
print(f)

f.popleft()
# you can now remove left or right of the list
print(f)

# to rotate list
f.rotate(1)
print(f)
