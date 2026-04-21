from datetime import datetime, time
from typing import Optional, Dict, List
from sqlmodel import Session, select
from database import BusinessHours, Department
from config import settings


class BusinessHoursManager:
    """Manages business hours and department availability."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def is_business_hours(self, department: str = None) -> bool:
        """
        Check if it's currently business hours.
        
        Args:
            department: Optional department to check specific hours
        
        Returns:
            True if within business hours
        """
        now = datetime.now()
        current_day = now.weekday()  # 0=Monday, 6=Sunday
        current_time = now.time()
        
        # Get business hours for today
        business_hours = self.session.exec(
            select(BusinessHours).where(BusinessHours.day_of_week == current_day)
        ).first()
        
        if not business_hours or business_hours.is_closed:
            return False
        
        # Parse time strings
        open_time = datetime.strptime(business_hours.open_time, "%H:%M").time()
        close_time = datetime.strptime(business_hours.close_time, "%H:%M").time()
        
        return open_time <= current_time < close_time
    
    def get_department_status(self, department: str) -> Dict[str, any]:
        """
        Get the status of a specific department.
        
        Args:
            department: Department name
        
        Returns:
            Dictionary with status information
        """
        is_open = self.is_business_hours(department)
        
        return {
            "department": department,
            "is_available": is_open,
            "message": f"{department} department is {'available' if is_open else 'closed'}"
        }
    
    def get_all_department_status(self) -> List[Dict[str, any]]:
        """Get status of all departments."""
        departments = [dept.value for dept in Department]
        return [self.get_department_status(dept) for dept in departments]
    
    def initialize_default_hours(self):
        """Initialize default business hours (9AM-6PM, Mon-Fri)."""
        # Check if hours already exist
        existing = self.session.exec(select(BusinessHours)).first()
        if existing:
            return
        
        # Create default hours for Monday-Friday
        for day in range(5):  # 0-4 (Mon-Fri)
            hours = BusinessHours(
                day_of_week=day,
                open_time="09:00",
                close_time="18:00",
                is_closed=False
            )
            self.session.add(hours)
        
        # Weekend closed
        for day in range(5, 7):  # 5-6 (Sat-Sun)
            hours = BusinessHours(
                day_of_week=day,
                open_time="09:00",
                close_time="18:00",
                is_closed=True
            )
            self.session.add(hours)
        
        self.session.commit()


class CallRouter:
    """Handles call routing logic."""
    
    DEPARTMENT_PHONE_MAP = {
        Department.SALES: "+919999999999",  # Replace with actual numbers
        Department.SUPPORT: "+919999999998",
        Department.BILLING: "+919999999997",
        Department.HR: "+919999999996",
        Department.GENERAL: "+919999999995"
    }
    
    def __init__(self, session: Session):
        self.session = session
        self.hours_manager = BusinessHoursManager(session)
    
    def route_call(self, department: str) -> Dict[str, any]:
        """
        Determine how to route a call based on department availability.
        
        Args:
            department: Target department
        
        Returns:
            Routing decision with phone number or voicemail instruction
        """
        status = self.hours_manager.get_department_status(department)
        
        if status["is_available"]:
            # Route to department phone
            phone_number = self.DEPARTMENT_PHONE_MAP.get(
                Department(department.lower()),
                self.DEPARTMENT_PHONE_MAP[Department.GENERAL]
            )
            return {
                "action": "transfer",
                "department": department,
                "phone_number": phone_number,
                "message": f"Transferring to {department} department"
            }
        else:
            # Route to voicemail
            return {
                "action": "voicemail",
                "department": department,
                "message": f"{department} department is closed. Taking a message."
            }
    
    def get_voicemail_greeting(self) -> str:
        """Get appropriate voicemail greeting based on time."""
        if self.hours_manager.is_business_hours():
            return "All our agents are currently busy. Please leave a message and we'll call you back shortly."
        else:
            return "We are currently closed. Please leave a message and we'll get back to you during business hours (9AM-6PM, Monday-Friday)."
