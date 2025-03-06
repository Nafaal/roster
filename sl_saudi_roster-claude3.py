import pandas as pd
import calendar
import datetime
import random
from tabulate import tabulate
import numpy as np

def generate_roster(start_year, start_month, end_year, end_month, sl_team_size, saudi_team_size,
                    shift_requirements):
    
    # Create list of all days in the specified date range
    start_date = datetime.date(start_year, start_month, 1)
    if end_month == 12:
        end_date = datetime.date(end_year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        end_date = datetime.date(end_year, end_month + 1, 1) - datetime.timedelta(days=1)
    
    # Create team members
    sl_team = [f"SL{i+1}" for i in range(sl_team_size)]
    saudi_team = [f"SA{i+1}" for i in range(saudi_team_size)]
    all_members = sl_team + saudi_team
    
    # Define working days
    sl_working_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}
    saudi_working_days = {'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'}
    
    # Initialize dates for roster
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += datetime.timedelta(days=1)
    
    # Initialize roster dataframe
    roster = pd.DataFrame(index=dates, columns=['Day', 'Day Shift', 'Evening Shift', 'Night Shift'])
    roster['Day'] = [calendar.day_name[date.weekday()] for date in roster.index]
    
    # Initialize assignment trackers with separate counters for each shift type
    assignment_count = {member: {'day': 0, 'evening': 0, 'night': 0, 'total': 0} for member in all_members}
    
    # Track last assigned shift day for each member to avoid consecutive night/day assignments
    # Now also tracking evening shifts which extend to next day
    last_assignment = {member: {'date': None, 'shift': None, 'evening_prior_day': False} for member in all_members}
    
    # Pre-calculate total requirements for the entire period to enable fair distribution
    total_required = {
        'SL': {'day': 0, 'evening': 0, 'night': 0},
        'Saudi': {'day': 0, 'evening': 0, 'night': 0}
    }
    
    for date in dates:
        day_name = calendar.day_name[date.weekday()]
        for team in ['SL', 'Saudi']:
            for shift in ['day', 'evening', 'night']:
                total_required[team][shift] += shift_requirements[day_name][team][shift]
    
    # Calculate target assignments per member for look-ahead planning
    target_assignments = {
        'SL': {
            'day': total_required['SL']['day'] / sl_team_size,
            'evening': total_required['SL']['evening'] / sl_team_size,
            'night': total_required['SL']['night'] / sl_team_size
        },
        'Saudi': {
            'day': total_required['Saudi']['day'] / saudi_team_size,
            'evening': total_required['Saudi']['evening'] / saudi_team_size,
            'night': total_required['Saudi']['night'] / saudi_team_size
        }
    }
    
    # Create dictionary to track daily shift assignments
    daily_shift_assignments = {date: set() for date in dates}
    
    # Track members who worked evening shift on previous day and should not be assigned the next day
    evening_shift_prior_day = set()
    
    # First pass: Create initial roster with balanced assignments and constraints
    for date, row in roster.iterrows():
        day_name = row['Day']
        day_req = shift_requirements[day_name]
        
        # Filter members by working days but make everyone available if needed
        sl_on_duty = sl_team if day_name in sl_working_days else []
        sl_off_duty = [] if day_name in sl_working_days else sl_team
        
        saudi_on_duty = saudi_team if day_name in saudi_working_days else []
        saudi_off_duty = [] if day_name in saudi_working_days else saudi_team
        
        # Reset daily assignment tracker for this date
        daily_shift_assignments[date] = set()
        
        # Anyone who worked evening shift yesterday cannot work today at all
        for member in evening_shift_prior_day:
            daily_shift_assignments[date].add(member)  # Mark as unavailable
        
        # Clear the evening shift set for today
        evening_shift_prior_day = set()
        
        # Make initial assignments for all three shifts in specific order: night, evening, day
        for shift_name, shift_key in [('Night Shift', 'night'), ('Evening Shift', 'evening'), ('Day Shift', 'day')]:
            sl_req = day_req['SL'][shift_key]
            saudi_req = day_req['Saudi'][shift_key]
            
            # Calculate score for each member based on multiple factors
            sl_scores = {}
            for member in sl_team:
                # Skip if already assigned to a shift today
                if member in daily_shift_assignments[date]:
                    continue
                    
                # Base score is inverse of current assignment count
                score = 100 - (assignment_count[member][shift_key] * 100 / max(1, target_assignments['SL'][shift_key]))
                
                # Penalty for off-duty days
                if member in sl_off_duty:
                    score -= 20
                
                # Penalty for consecutive shifts (especially day after night)
                if last_assignment[member]['date'] == date - datetime.timedelta(days=1):
                    score -= 10
                    if last_assignment[member]['shift'] == 'night' and shift_key == 'day':
                        score -= 50  # Very heavy penalty for night->day transition
                
                sl_scores[member] = score
            
            # Same for Saudi team
            saudi_scores = {}
            for member in saudi_team:
                # Skip if already assigned to a shift today
                if member in daily_shift_assignments[date]:
                    continue
                    
                score = 100 - (assignment_count[member][shift_key] * 100 / max(1, target_assignments['Saudi'][shift_key]))
                if member in saudi_off_duty:
                    score -= 20
                if last_assignment[member]['date'] == date - datetime.timedelta(days=1):
                    score -= 10
                    if last_assignment[member]['shift'] == 'night' and shift_key == 'day':
                        score -= 50  # Very heavy penalty for night->day transition
                
                saudi_scores[member] = score
            
            # Select members with highest scores
            sl_selected = sorted(sl_scores.items(), key=lambda x: x[1], reverse=True)[:sl_req]
            saudi_selected = sorted(saudi_scores.items(), key=lambda x: x[1], reverse=True)[:saudi_req]
            
            # If not enough members available, try to find more
            if len(sl_selected) < sl_req:
                # Try including off-duty members if needed
                additional_members = []
                for member in sl_off_duty:
                    if member not in daily_shift_assignments[date] and member not in [m[0] for m in sl_selected]:
                        additional_members.append((member, 0))  # Lower score for off-duty
                        if len(sl_selected) + len(additional_members) >= sl_req:
                            break
                
                sl_selected.extend(additional_members[:sl_req - len(sl_selected)])
            
            if len(saudi_selected) < saudi_req:
                # Try including off-duty members if needed
                additional_members = []
                for member in saudi_off_duty:
                    if member not in daily_shift_assignments[date] and member not in [m[0] for m in saudi_selected]:
                        additional_members.append((member, 0))  # Lower score for off-duty
                        if len(saudi_selected) + len(additional_members) >= saudi_req:
                            break
                
                saudi_selected.extend(additional_members[:saudi_req - len(saudi_selected)])
            
            # Combine and update assignments
            shift_members = [m[0] for m in sl_selected + saudi_selected]
            
            for member in shift_members:
                assignment_count[member][shift_key] += 1
                assignment_count[member]['total'] += 1
                last_assignment[member]['date'] = date
                last_assignment[member]['shift'] = shift_key
                daily_shift_assignments[date].add(member)  # Mark as assigned for today
                
                # If this is an evening shift, mark them as needing rest the next day
                if shift_key == 'evening':
                    evening_shift_prior_day.add(member)
            
            # Record in roster
            roster.at[date, shift_name] = ', '.join(shift_members) if shift_members else 'None'
    
    # Second pass: Optimize roster to balance assignments even better
    perform_balance_optimization(roster, sl_team, saudi_team, assignment_count, target_assignments)
    
    # Generate balance metrics
    balance_metrics = calculate_balance_metrics(assignment_count, sl_team, saudi_team)
    
    return roster, assignment_count, balance_metrics

def perform_balance_optimization(roster, sl_team, saudi_team, assignment_count, target_assignments):
    """Perform swaps to optimize balance between team members"""
    # Track daily assignments during optimization to prevent multi-shift assignment
    daily_assignments = {}
    
    # Also track evening shift assignments to prevent next-day assignments
    evening_assignments = {}
    
    # First, build our tracking dictionaries
    for date_idx, date in enumerate(roster.index):
        daily_assignments[date] = set()
        
        for shift_col in ['Day Shift', 'Evening Shift', 'Night Shift']:
            shift_members = roster.at[date, shift_col].split(', ') if roster.at[date, shift_col] != 'None' else []
            for member in shift_members:
                daily_assignments[date].add(member)
            
            # Track evening shift members specifically
            if shift_col == 'Evening Shift':
                evening_assignments[date] = set(shift_members)
    
    # Identify the most overworked and underworked team members
    for team, members in [('SL', sl_team), ('Saudi', saudi_team)]:
        for shift_type in ['day', 'evening', 'night']:
            # Sort by number of assignments for this shift type
            sorted_members = sorted(members, key=lambda m: assignment_count[m][shift_type])
            
            # Try up to 50 optimization iterations
            for _ in range(50):
                underworked = sorted_members[0]  # Least assigned
                overworked = sorted_members[-1]  # Most assigned
                
                # If difference is small, no need to balance
                if assignment_count[overworked][shift_type] - assignment_count[underworked][shift_type] <= 1:
                    break
                
                # Find a date where we can swap these members
                swap_made = False
                for date_idx, date in enumerate(roster.index):
                    shift_column = None
                    if shift_type == 'day':
                        shift_column = 'Day Shift'
                    elif shift_type == 'evening':
                        shift_column = 'Evening Shift'
                    else:
                        shift_column = 'Night Shift'
                    
                    shift_members = roster.at[date, shift_column].split(', ') if roster.at[date, shift_column] != 'None' else []
                    
                    # If overworked member is working and underworked is not
                    if overworked in shift_members and underworked not in shift_members:
                        # Check if underworked has another shift on this day
                        if underworked in daily_assignments[date]:
                            continue
                        
                        # For evening shifts, check if the underworked member has a shift next day
                        if shift_column == 'Evening Shift' and date_idx < len(roster.index) - 1:
                            next_day = roster.index[date_idx + 1]
                            if underworked in daily_assignments.get(next_day, set()):
                                continue
                        
                        # For any shift, check if underworked worked evening shift previous day
                        if date_idx > 0:
                            prev_day = roster.index[date_idx - 1]
                            if underworked in evening_assignments.get(prev_day, set()):
                                continue
                            
                        # Perform the swap
                        shift_members.remove(overworked)
                        shift_members.append(underworked)
                        roster.at[date, shift_column] = ', '.join(shift_members)
                        
                        # Update assignment counts
                        assignment_count[overworked][shift_type] -= 1
                        assignment_count[overworked]['total'] -= 1
                        assignment_count[underworked][shift_type] += 1
                        assignment_count[underworked]['total'] += 1
                        
                        # Update daily assignments tracking
                        daily_assignments[date].remove(overworked)
                        daily_assignments[date].add(underworked)
                        
                        # Update evening shift tracking if needed
                        if shift_column == 'Evening Shift':
                            evening_assignments[date].remove(overworked)
                            evening_assignments[date].add(underworked)
                        
                        swap_made = True
                        break
                
                # If no swap was possible, move to the next iteration
                if not swap_made:
                    break
                
                # Re-sort after swap
                sorted_members = sorted(members, key=lambda m: assignment_count[m][shift_type])

def calculate_balance_metrics(assignment_count, sl_team, saudi_team):
    """Calculate metrics that quantify the balance of the roster"""
    metrics = {}
    
    # Calculate metrics for each team and shift type
    for team_name, team in [("Sri Lanka", sl_team), ("Saudi Arabia", saudi_team)]:
        team_metrics = {}
        
        for shift_type in ['day', 'evening', 'night', 'total']:
            counts = [assignment_count[member][shift_type] for member in team]
            team_metrics[shift_type] = {
                'min': min(counts),
                'max': max(counts),
                'mean': sum(counts) / len(counts),
                'std_dev': np.std(counts),
                'fairness_score': 100 * (1 - (max(counts) - min(counts)) / max(1, max(counts)))
            }
        
        metrics[team_name] = team_metrics
    
    return metrics

def main():
    print("Roster Planning System for Sri Lanka and Saudi Arabia Teams")
    print("--------------------------------------------------------")
    
    # Get input parameters
    try:
        start_year = int(input("Start Year: "))
        start_month = int(input("Start Month (1-12): "))
        end_year = int(input("End Year: "))
        end_month = int(input("End Month (1-12): "))
        
        sl_team_size = int(input("Number of team members in Sri Lanka: "))
        saudi_team_size = int(input("Number of team members in Saudi Arabia: "))
        
        print("\nShift Requirements (Enter number of people for each shift):")
        
        # Initialize shift requirements dictionary
        shift_requirements = {
            'Monday': {'SL': {'day': 0, 'evening': 0, 'night': 0}, 'Saudi': {'day': 0, 'evening': 0, 'night': 0}},
            'Tuesday': {'SL': {'day': 0, 'evening': 0, 'night': 0}, 'Saudi': {'day': 0, 'evening': 0, 'night': 0}},
            'Wednesday': {'SL': {'day': 0, 'evening': 0, 'night': 0}, 'Saudi': {'day': 0, 'evening': 0, 'night': 0}},
            'Thursday': {'SL': {'day': 0, 'evening': 0, 'night': 0}, 'Saudi': {'day': 0, 'evening': 0, 'night': 0}},
            'Friday': {'SL': {'day': 0, 'evening': 0, 'night': 0}, 'Saudi': {'day': 0, 'evening': 0, 'night': 0}},
            'Saturday': {'SL': {'day': 0, 'evening': 0, 'night': 0}, 'Saudi': {'day': 0, 'evening': 0, 'night': 0}},
            'Sunday': {'SL': {'day': 0, 'evening': 0, 'night': 0}, 'Saudi': {'day': 0, 'evening': 0, 'night': 0}}
        }
        
        # Sri Lanka shifts (Mon-Fri)
        print("\nSri Lanka Shifts (Mon-Fri):")
        sl_weekday_day = int(input("Mon-Fri Day Shift count: "))
        sl_weekday_evening = int(input("Mon-Fri Evening Shift count: "))
        sl_weekday_night = int(input("Mon-Fri Night Shift count: "))
        
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            shift_requirements[day]['SL']['day'] = sl_weekday_day
            shift_requirements[day]['SL']['evening'] = sl_weekday_evening
            shift_requirements[day]['SL']['night'] = sl_weekday_night
        
        # Sri Lanka Saturday shifts
        print("\nSri Lanka Saturday Shifts:")
        shift_requirements['Saturday']['SL']['day'] = int(input("Saturday Day Shift count: "))
        shift_requirements['Saturday']['SL']['evening'] = int(input("Saturday Evening Shift count: "))
        shift_requirements['Saturday']['SL']['night'] = int(input("Saturday Night Shift count: "))
        
        # Sri Lanka Sunday shifts
        print("\nSri Lanka Sunday Shifts:")
        shift_requirements['Sunday']['SL']['day'] = int(input("Sunday Day Shift count: "))
        shift_requirements['Sunday']['SL']['evening'] = int(input("Sunday Evening Shift count: "))
        shift_requirements['Sunday']['SL']['night'] = int(input("Sunday Night Shift count: "))
        
        # Saudi shifts (Sun-Thu)
        print("\nSaudi Arabia Shifts (Sun-Thu):")
        saudi_weekday_day = int(input("Sun-Thu Day Shift count: "))
        saudi_weekday_evening = int(input("Sun-Thu Evening Shift count: "))
        saudi_weekday_night = int(input("Sun-Thu Night Shift count: "))
        
        for day in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']:
            shift_requirements[day]['Saudi']['day'] = saudi_weekday_day
            shift_requirements[day]['Saudi']['evening'] = saudi_weekday_evening
            shift_requirements[day]['Saudi']['night'] = saudi_weekday_night
        
        # Saudi Friday shifts
        print("\nSaudi Arabia Friday Shifts:")
        shift_requirements['Friday']['Saudi']['day'] = int(input("Friday Day Shift count: "))
        shift_requirements['Friday']['Saudi']['evening'] = int(input("Friday Evening Shift count: "))
        shift_requirements['Friday']['Saudi']['night'] = int(input("Friday Night Shift count: "))
        
        # Saudi Saturday shifts
        print("\nSaudi Arabia Saturday Shifts:")
        shift_requirements['Saturday']['Saudi']['day'] = int(input("Saturday Day Shift count: "))
        shift_requirements['Saturday']['Saudi']['evening'] = int(input("Saturday Evening Shift count: "))
        shift_requirements['Saturday']['Saudi']['night'] = int(input("Saturday Night Shift count: "))
        
        # Validate inputs
        if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
            raise ValueError("Month must be between 1 and 12")
        
        if datetime.date(start_year, start_month, 1) > datetime.date(end_year, end_month, 1):
            raise ValueError("Start date must be before end date")
        
        # Summary of requirements
        print("\nShift Requirements Summary:")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days:
            print(f"\n{day}:")
            for team in ['SL', 'Saudi']:
                print(f"  {team} Team: Day: {shift_requirements[day][team]['day']}, "
                      f"Evening: {shift_requirements[day][team]['evening']}, "
                      f"Night: {shift_requirements[day][team]['night']}")
        
        # Generate roster with improved balance
        roster, assignment_count, balance_metrics = generate_roster(
            start_year, start_month, end_year, end_month, 
            sl_team_size, saudi_team_size,
            shift_requirements
        )
        
        # Print the roster in a nice table format
        print("\nGenerated Roster:")
        
        # Format roster for display
        roster_display = roster.copy()
        roster_display.index = [idx.strftime('%Y-%m-%d') for idx in roster_display.index]
        
        print(tabulate(roster_display, headers='keys', tablefmt='grid'))
        
        # Print balance statistics
        print("\n=== Distribution Fairness Report ===")
        
        for team, team_members in [("Sri Lanka", [f"SL{i+1}" for i in range(sl_team_size)]), 
                                 ("Saudi Arabia", [f"SA{i+1}" for i in range(saudi_team_size)])]:
            print(f"\n{team} Team Distribution:")
            
            # Print shift-specific metrics
            for shift_type in ['day', 'evening', 'night', 'total']:
                metrics = balance_metrics[team][shift_type]
                print(f"\n  {shift_type.capitalize()} Shift Metrics:")
                print(f"    Min: {metrics['min']}, Max: {metrics['max']}, Mean: {metrics['mean']:.2f}")
                print(f"    Standard Deviation: {metrics['std_dev']:.2f}")
                print(f"    Fairness Score (100 is perfect): {metrics['fairness_score']:.2f}")
            
            # Print detailed per-member breakdown
            print("\n  Detailed Assignment Counts:")
            member_data = []
            for member in team_members:
                member_data.append([
                    member,
                    assignment_count[member]['day'],
                    assignment_count[member]['evening'],
                    assignment_count[member]['night'],
                    assignment_count[member]['total']
                ])
            
            # Sort by total assignments
            member_data.sort(key=lambda x: x[4], reverse=True)
            
            print(tabulate(member_data, 
                         headers=['Member', 'Day Shifts', 'Evening Shifts', 'Night Shifts', 'Total'],
                         tablefmt='grid'))
        
        # Save to CSV
        filename = f"balanced_roster_{start_year}_{start_month}_to_{end_year}_{end_month}.csv"
        roster.to_csv(filename)
        print(f"\nRoster saved to {filename}")
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
