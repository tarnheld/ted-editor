import tkinter as tk
import sys
class FSM:
  def __init__(self, initialState, transitions, widget):
    self._state  = initialState
    self._tt     = transitions
    self._widget = widget
    self._events = self.unique_events()
    self._stopped = True
    self.start()
  def unique_events(self):
      evs = set()
      # find unique events
      for state, tag, event in self._tt.keys():
        if event is not None: # None is any event
          evs.add((tag,event))
      return evs
  def start(self):
    for tag,event in self._events:
      # event thunk that saves the event info in extra arguments
      def _trans(ev,self = self,te = (tag,event)):
        return self.transition(ev,te[0],te[1])
      if (tag):
        self._widget.tag_bind(tag,event,_trans)
      else:
        self._widget.bind(event,_trans)
    self._stopped = False
  def stop(self):
    # bind all unique events
    for tag,event in self._events:
      if (tag):
        self._widget.tag_unbind(tag,event)
      else:
        self._widget.unbind(event)
    self._stopped = True

  def isStopped(self):
    return self._stopped
 
  def transition(self,ev,tag,event):
    #print ("transition from state",self._state,tag,event)
    tr = None
    key = (self._state,tag,event)
    if tk.CURRENT:
      tags = ev.widget.gettags(tk.CURRENT)
      if not tag in tags or not key in self._tt:
        #print ("no tags transition found",key)
        key = (self._state,None,event)
    if not key in self._tt:
      # check for any event transition
      key = (self._state,None,None)
    if key in self._tt:
      tr = self._tt[key]
      #print ("transition found:",key,tr[0])
      if tr[1]: # callback
        tr[1](ev)
      self._state = tr[0] # set new state
    else:
      #print ("no transition found:", self._state, tag, event)
      pass
    sys.stdout.flush()
