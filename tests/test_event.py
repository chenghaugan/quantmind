"""异步事件引擎测试。"""
import asyncio

from quantmind.core import EventEngine, Event, EventType


async def test_event_dispatch():
    eng = EventEngine()
    received = []

    def handler(event: Event) -> None:
        received.append(event.data)

    eng.register(EventType.EVENT_BAR, handler)
    eng.put(Event(EventType.EVENT_BAR, "bar1"))
    eng.put(Event(EventType.EVENT_BAR, "bar2"))

    await eng.start()
    await asyncio.sleep(0.1)
    await eng.stop()

    assert received == ["bar1", "bar2"]


async def test_general_handler():
    eng = EventEngine()
    seen = []

    eng.register_general(lambda e: seen.append(e.type))
    eng.put(Event(EventType.EVENT_TICK, 1))
    await eng.start()
    await asyncio.sleep(0.1)
    await eng.stop()
    assert EventType.EVENT_TICK in seen
