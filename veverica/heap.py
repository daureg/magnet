"""The C-implementation of heapq runs *very* fast.

However, heapq doesn't support decrease-key/increase-key.

The heap class here is a simple wrapper around heapq with support for
updating the priority as suggested in the doc for heapq.
"""

from collections import MutableMapping

class heap(MutableMapping):
	"""binary min-heap based on heapq
	"""
	
	_REMOVED = object() # a sentinel that indicates that an entry has been removed.
	
	def __init__(self,items):
		"""Creates a heap from a Mapping (like a dict or another heap) or an iterable.
		"""
		from heapq import heapify
		from itertools import count
		from collections import Mapping # requires Python 2.6+
		
		counter = count()
		
		if isinstance(items,Mapping):
			# Before you get all upset about typechecking in Python,
			# recall that Python's dict behaves this way too.
			# dict(x) behaves differently depending on whether x is a Mapping 
			# (like another dict) or just a plain old iterable
			# (like a tuple of tuples).
			self._entries = [[priority, next(counter), item] for item, priority in items.items()]
		
		else:
			# If items is not a mapping it is a plain old iterable.
			# In this case, we set the priority to be the item itself.
			self._entries = [[item, next(counter), item] for item in items]
			
		self._entry_finder = {entry[-1]:entry for entry in self._entries}
		self._counter = counter
		
		heapify(self._entries)
	
	def __getitem__(self,item):
		"""Returns the priority of given item
		"""
		return self._entry_finder[item][0]
	
	def __setitem__(self,item,priority):
		"""Sets an item to the given priority.
		If the item is not yet in this heap, it is inserted.
		"""
		from heapq import heappush
		
		if item in self._entry_finder:
			del self[item]
		
		entry = [priority, next(self._counter), item]
		self._entry_finder[item] = entry
		
		heappush(self._entries, entry)
	
	def __delitem__(self,item):
		"""Removes an item from this heap.
		Raises KeyError if item is not in this heap.
		"""
		# Here we simply mark an item as removed.
		# The item is actually removed from self._entries lazily as needed 
		# when `pop()`, `peek()` or `__repr__()` is is called.
		self._entry_finder.pop(item)[-1] = self._REMOVED
	
	def __iter__(self):
		"""Returns each item in no particular order.
		
		WARNING: modifying the heap while iterating over it may result in strange behavior!
		"""
		return (item for _, _, item in self._entries if item is not self._REMOVED)

	def __len__(self):
		"""Returns the number of items still in this heap
		"""
		return len(self._entry_finder)
		
	def __contains__(self,item):
		"""Returns True if item is still in this heap
		"""
		return item in self._entry_finder
	
	def __repr__(self):
		"""Returns a string of the contents of the heap in order of underlying heap array.
		"""
		from heapq import heapify
		
		if len(self._entry_finder) < len(self._entries):
			# If '_entries' has more elements than '_entry_finder',
			# then there are still lingering '_REMOVED' in '_entries'.
			# since __repr__ is going to take time linear in the number of
			# elements anyway, it seems appropriate to do some cleanup.
			self._entries = [entry for entry in self._entries if entry[-1] is not self._REMOVED]
			heapify(self._entries)
		
		return 'heap({%s})'%', '.join('%s: %s'%(item,priority) for priority, _, item in self._entries)
	
	def keys(self):
		"""Returns an iterable of items in self. Identical to iter(self)
		
		WARNING: modifying the heap while iterating over it may result in strange behavior!
		"""
		return iter(self)
	
	def values(self):
		"""Returns an iterable of priorities in self.
		
		WARNING: modifying the heap while iterating over it may result in strange behavior!
		"""
		return (priority for priority, _, item in self._entries if item is not self._REMOVED)
	
	def pop(self):
		"""Pops item with smallest priority value.
		If there are ties, the first element inserted is removed first.
		"""
		from heapq import heappop
		
		while self._entries:
			_, _, item = heappop(self._entries)
			if item is not self._REMOVED:
				del self._entry_finder[item]
				return item
		raise KeyError('pop from an empty heap')
	
	def popitem(self):
		"""Pops an item together with its priority.
		"""
		from heapq import heappop
		
		while self._entries:
			priority, _, item = heappop(self._entries)
			if item is not self._REMOVED:
				del self._entry_finder[item]
				return item, priority
		raise KeyError('popitem from an empty heap')
	
	def peek(self):
		"""Finds the item with smallest priority value like pop, but does not remove the item.
		"""
		from heapq import heappop
		while self._entries:
			_, _, item = self._entries[0]
			if item is not self._REMOVED:
				return item
			heappop(self._entries)
		raise KeyError('peek at an empty heap')
	
	def peekitem(self):
		"""Like peek, but also returns the priority
		"""
		from heapq import heappop
		while self._entries:
			priority, _, item = self._entries[0]
			if item is not self._REMOVED:
				return item, priority
			heappop(self._entries)
		raise KeyError('peekitem at an empty heap')
	
	def add(self,item,priority=None):
		"""Adds an item with given priority.
		If the item already exists in the heap, its priority is updated.
		
		This method is equivalent to:
		
			self[item] = item if priority is None else priority
		"""
		self[item] = item if priority is None else priority
	
	def discard(self,item):
		"""Removes item from this heap if it exists.
		"""
		if item in self:
			del self[item]

