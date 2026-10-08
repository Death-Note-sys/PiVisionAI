with open('app/core/coordinator.py', 'r') as f:
    coord = f.read()
if 'maxsize' not in coord:
    coord = coord.replace('self.event_bus = event_bus', 'self.event_bus = event_bus\n        import queue\n        self.command_queue = queue.Queue(maxsize=100)')
    with open('app/core/coordinator.py', 'w') as f:
        f.write(coord)
