# TODO: Try using json stuff? Store transcripts in db so later Verifiable by doctor, grep when run, run bash script to start
import re
import sqlite3

import nltk

import random
from datetime import datetime, timedelta


def random_datetime():
    """Generates a random date and time

    This function generates a random date that is:
    - More than 2 weeks but less than a month from the current date, Monday to Friday.
    - Has a time between 9:00 and 17:00 in 15-minute increments (9:00, 9:15, ..., 16:45).

    Returns:
        A datetime object representing the randomly generated date and time.
    """

    # Get the current date
    now = datetime.now()

    # Set the minimum and maximum date range
    min_date = now + timedelta(days=14)
    max_date = now + timedelta(days=30)

    # Generate a random date within the specified range
    delta = max_date - min_date
    random_days = random.randrange(delta.days)
    random_date = min_date + timedelta(days=random_days)

    # Ensure the date is Monday to Friday
    weekdays = [0, 1, 2, 3, 4]  # 0 is Monday
    while random_date.weekday() not in weekdays:
        random_days += 1
        random_date = min_date + timedelta(days=random_days)

    # Generate acceptable times (9:00, 9:15, ..., 16:45)
    times = [f"{i:02d}:{j:02d}" for i in range(9, 18) for j in (0, 15)]

    # Pick a random time among the acceptable ones
    random_time = random.choice(times)

    # Combine the date and time into a datetime object
    datetime_str = f"{random_date.strftime('%Y-%m-%d')} {random_time}"
    return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")


def extract_time(user_input):
    """ Use re to look for the time described
    :param user_input: Presumes user is responding to prompt
    :return:
    """

    time_pattern = r'\b(0?[1-9]|1[0-2])(:[0-5][0-9])?\s?([ap]m)\b'
    time_matches = list(re.finditer(time_pattern, user_input, re.IGNORECASE))
    if time_matches:
        # Returns the longest match
        time_matches.sort(key=lambda match: len(match.group(0)), reverse=True)
        booking_time = time_matches[0].group(0)
        return booking_time
    else:
        return None

def insert_into_database(name, time,day, notes):
    try:
        connection = sqlite3.connect('booking.db')
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS appointments
                        (names text NOT NULL, start_time text NOT NULL, days text NOT NULL, notes text NOT NULL);''')

        cursor.execute("INSERT INTO appointments VALUES (?,?,?,?)", (name, time, day,notes))
        task1 = cursor.execute("SELECT * FROM bookings")

        connection.commit()
    except sqlite3.Error as error:
        # Handle the error
        print(f"Error: {error}")
        errored = True
    else:
        # print("Booking added to database:", task1.fetchall())
        errored = False

    finally:
        if connection:
            connection.close()

    return errored

def make_appointment(user_input, name, notes = ''):
    """ Prompts user until they provide valid booking information. Inserts the data into the database.
    :param user_input: what the user said
    :param name: the users name
    """

    date_time = random_datetime()
    day = date_time.date()
    day = day.strftime('%Y-%m-%d')

    print("Neutrino: Is the ", day ,"at",date_time.time()," a good time for you? Yes/No")
    user_input = input('%s: ' % name)
    if user_input.lower() == 'no':
        appointment_time = None
        print("Neutrino: My apologies, %s. I can do a different time on that day." % name)
    else:
        appointment_time = date_time.time()
        appointment_time = appointment_time.strftime('%H:%M')

    while appointment_time is None:
        print("Neutrino: What time would you like the appointment for? I'll see if it's available")
        user_input = input('%s: ' % name)
        appointment_time = extract_time(user_input)
        if appointment_time is None:
            print(
                "Neutrino: I'm sorry, I didn't quite catch the time for your booking. Try specifying the time of day with"
                " a 'am' or 'pm'.")

    print(f"Neutrino: You would like to make a booking with these details:\n"
          f"   Time: {appointment_time} \n"
          f"   Date: {day}\n "
          f"If these details are correct, please say yes.")
    confirmed = input('%s: ' % name).lower()

    if confirmed == 'yes':

        insert_into_database(name, appointment_time, day, notes)
        print("Neutrino: Your appointment with a GP has been made, and I've sent along the notes from this consultation. We look forward to seeing you.")
    else:
        print("Neutrino: My apologies, %s. Lets try this again." % name)
        make_appointment(' ', name)

