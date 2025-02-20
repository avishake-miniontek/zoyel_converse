"""
Timer management functions for the LLM to use.
Provides capabilities to create, list, and manage timers.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import threading
import asyncio
import uuid
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Timer:
    id: str
    name: str
    duration: float  # in seconds
    start_time: datetime
    end_time: datetime
    completed: bool = False

class TimerManager:
    """Singleton manager for all timers"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.timers: Dict[str, Timer] = {}
                cls._instance.tasks: Dict[str, asyncio.Task] = {}
        return cls._instance
    
    async def _timer_callback(self, timer_id: str):
        """Callback when timer completes"""
        try:
            await asyncio.sleep(self.timers[timer_id].duration)
            self.timers[timer_id].completed = True
            logger.info(f"Timer {timer_id} ({self.timers[timer_id].name}) completed")
        except asyncio.CancelledError:
            logger.info(f"Timer {timer_id} cancelled")
        except Exception as e:
            logger.error(f"Error in timer callback: {e}")
        finally:
            if timer_id in self.tasks:
                del self.tasks[timer_id]

# Global timer manager instance
timer_manager = TimerManager()

def set_timer(
    duration: float,
    name: Optional[str] = None,
) -> Dict[str, any]:
    """
    Create a new timer with the specified duration.
    
    Args:
        duration: Time in seconds for the timer
        name: Optional name/description for the timer
        
    Returns:
        Dictionary containing timer details
    """
    try:
        # Generate timer ID and set name
        timer_id = str(uuid.uuid4())
        timer_name = name or f"Timer {timer_id[:8]}"
        
        # Create timer object
        now = datetime.now()
        timer = Timer(
            id=timer_id,
            name=timer_name,
            duration=duration,
            start_time=now,
            end_time=now + timedelta(seconds=duration)
        )
        
        # Store timer
        timer_manager.timers[timer_id] = timer
        
        # Create async task for timer
        loop = asyncio.get_event_loop()
        task = loop.create_task(timer_manager._timer_callback(timer_id))
        timer_manager.tasks[timer_id] = task
        
        return {
            "id": timer_id,
            "name": timer_name,
            "duration": duration,
            "end_time": timer.end_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error setting timer: {e}")
        raise

def list_timers() -> List[Dict[str, any]]:
    """
    List all active timers.
    
    Returns:
        List of dictionaries containing timer details
    """
    try:
        active_timers = []
        now = datetime.now()
        
        for timer in timer_manager.timers.values():
            if not timer.completed:
                remaining = (timer.end_time - now).total_seconds()
                if remaining > 0:
                    active_timers.append({
                        "id": timer.id,
                        "name": timer.name,
                        "remaining_seconds": remaining,
                        "end_time": timer.end_time.isoformat()
                    })
                    
        return active_timers
        
    except Exception as e:
        logger.error(f"Error listing timers: {e}")
        raise

def cancel_timer(timer_id: str) -> Dict[str, any]:
    """
    Cancel a specific timer.
    
    Args:
        timer_id: ID of the timer to cancel
        
    Returns:
        Dictionary containing result of cancellation
    """
    try:
        if timer_id not in timer_manager.timers:
            raise ValueError(f"Timer {timer_id} not found")
            
        timer = timer_manager.timers[timer_id]
        if timer.completed:
            return {
                "success": False,
                "message": f"Timer {timer_id} already completed"
            }
            
        # Cancel the async task if it exists
        if timer_id in timer_manager.tasks:
            timer_manager.tasks[timer_id].cancel()
            
        timer.completed = True
        
        return {
            "success": True,
            "message": f"Timer {timer_id} ({timer.name}) cancelled successfully"
        }
        
    except Exception as e:
        logger.error(f"Error cancelling timer: {e}")
        raise

def register(registry):
    """Register timer functions with the function registry."""
    registry.register_function(set_timer)
    registry.register_function(list_timers)
    registry.register_function(cancel_timer)