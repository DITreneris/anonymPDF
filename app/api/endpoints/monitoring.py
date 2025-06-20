from fastapi import APIRouter, Depends
from typing import List, Dict, Any

from app.core.real_time_monitor import RealTimeMonitor, get_real_time_monitor

router = APIRouter()

@router.get("/status", response_model=Dict[str, Any])
def get_monitoring_status(monitor: RealTimeMonitor = Depends(get_real_time_monitor)):
    """
    Provides a summary of the application's real-time performance,
    including averages for duration, CPU, and memory usage.
    """
    return monitor.get_summary()

@router.get("/events", response_model=List[Dict[str, Any]])
def get_monitoring_events(limit: int = 100, monitor: RealTimeMonitor = Depends(get_real_time_monitor)):
    """
    Retrieves a list of the latest raw performance events recorded by the monitor.
    """
    return monitor.get_latest_metrics(limit=limit) 