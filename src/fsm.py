import tkinter as tk
import sys
from copy import deepcopy
import collections
import types


def _isiterable(x):
    return isinstance(x, collections.Iterable) and not isinstance(x,types.StringTypes)

class FSM:
    """a finite state machine bas class to ease tkinter bindings"""
    def __init__(self, initialState, transitions, widget):
        self._state = initialState
        self._tt = transitions
        self._widget = widget
        self._events = self.unique_events()
        self._stopped = True
        self.start()
    def unique_events(self):
        evs = set()
        # find unique events
        for state, tag, event in self._tt.keys():
            if event is not None:  # None is any event
                evs.add((tag, event))
        return evs
    def start(self):
        for tag, event in self._events:
            # event thunk that saves the event info in extra arguments
            def _trans(ev, self=self, te=(tag, event)):
                return self.transition(ev, te[0], te[1])
            if (tag):
                self._widget.tag_bind(tag, event, _trans)
            else:
                self._widget.bind(event, _trans)
        self._stopped = False
    def stop(self):
        # bind all unique events
        for tag, event in self._events:
            if (tag):
                self._widget.tag_unbind(tag, event)
            else:
                self._widget.unbind(event)
        self._stopped = True

    def isStopped(self):
        return self._stopped

    def transition(self, ev, tag, event):
        # print ("transition from state",self._state,tag,event)
        tr = None
        key = (self._state, tag, event)
        if tk.CURRENT:
            tags = ev.widget.gettags(tk.CURRENT)
            if not tag in tags or not key in self._tt:
                # print ("no tags transition found",key)
                key = (self._state, None, event)
        if not key in self._tt:
            # check for any event transition
            key = (self._state, None, None)
        if key in self._tt:
            new_state, cbs = self._tt[key]
            # print ("transition found:",key,tr[0])
            if cbs:  # callback
                if _isiterable(cbs):
                    for cb in cbs:
                        cb(ev)
                else:
                    cbs(ev)
            self._state = new_state  # set new state
        else:
            # print ("no transition found:", self._state, tag, event)
            pass
        sys.stdout.flush()

class SavedFSM(FSM):
    def __init__(self, initialState, transitions, widget,
                 undobinding="<Control-Key-z>",
                 redobinding="<Control-Key-y>"):
        undo_redo = {
            (initialState, None, redobinding): (initialState, self.onRedo),
            (initialState, None, undobinding): (initialState, self.onUndo),
        }

        transitions.update(undo_redo)

        super().__init__(initialState, transitions, widget)

        self.history = []
        self.future = []

    def historySave(self):
        self.history.append(deepcopy(self.save()))  # save copy of control curve
        self.future = []  # clear redos
        print("Save", len(self.future), len(self.history))
    def historyClear(self):
        self.history = []  # clear undos
        self.future  = []  # clear redos
    def historyTakeover(self,other):
        self.history.extend(other.history)
        self.future  = []  # clear redos
        self.restore(other.save())
        print("takeover", len(self.future), len(self.history))
    def save(self):
        print("please implement save in your derived class!!!")
    def restore(self, data):
        print("please implement restore in your derived class!!!")
    def onUndo(self, ev):
        if self.history:  # not empty
            current = self.history.pop()
            self.future.append(deepcopy(self.save()))
            self.restore(current)
        print("Undo", len(self.future), len(self.history))
    def onRedo(self, ev):
        if self.future:  # not empty
            current = self.future.pop()
            self.history.append(deepcopy(self.save()))
            self.restore(current)
        print("Redo", len(self.future), len(self.history))
