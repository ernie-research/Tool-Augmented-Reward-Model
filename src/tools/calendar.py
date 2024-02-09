import calendar
from datetime import datetime, timedelta
from typing import Any

class Calendar:

    # 1. get now time
    def now_time(self):
        return datetime.now().strftime('%Y-%m-%d')
    
    # 2. weekday
    def week_day(self, date: str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
        return calendar.day_name[date.weekday()]

    def target_day(self, theDay, diff):
        theDay = datetime.strptime(theDay, "%Y-%m-%d").date()
        if diff < 0:
            targetDay = theDay - timedelta(-diff)
        else:
            targetDay = theDay + timedelta(diff)
        return targetDay.strftime("%Y-%m-%d") 
    
    def day_difference(self, date1, date2):
        date1 = datetime.strptime(date1, "%Y-%m-%d").date()
        date2 = datetime.strptime(date2, "%Y-%m-%d").date()
        return abs((date1 - date2).days)
    
    def __call__(self, query) -> Any:
        split_query = query.split(', ')
        if len(split_query) == 1:
            return self.week_day()
            