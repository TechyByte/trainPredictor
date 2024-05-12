import datetime

import MySQLdb

import config

from schedule.scheduled_service import ScheduledService

# TODO: Get scheduled services

# TODO: Filter scheduled services based on query parameters
# TODO: Build SQL query based on the query parameters
# Second tiploc is optional and means an edge should be queried instead of a node (bidirectional)

# TODO: Return the filtered scheduled services
db = MySQLdb.connect(host=config.db_host, user=config.db_user, passwd=config.db_password, db=config.db_database)



def get_scheduled_services(date, tiploc, origin=False, destination=False, train_service_code=None,
                           also_passing_tiploc=None) -> [ScheduledService]:
    c = db.cursor()
    services = []
    two_letter_day = date.strftime("%A")[:2].lower()
    if origin:
        origin_supp = " AND WHERE `order` = 1"
    else:
        origin_supp = ""
    if destination:
        destination_supp = " AND WHERE `order` IN (SELECT `order` FROM `location` WHERE "
    else:
        destination_supp = ""
    standard_initial_query = f"SELECT id, train_uid, train_identity, train_service_code, atoc_code FROM `schedule` WHERE `start_date` <= '{date}' AND `end_date` >= '{date}' AND `runs_{two_letter_day}` = 1 AND `id` IN (SELECT id FROM `location` WHERE `tiploc_code` = '{tiploc}'{origin_supp}{destination_supp});"
    # returns all services passing through a tiploc on the given date

    # To make this query only an origin, also specify that orderID is 1 in query
    # To make this query only a destination, also specify that orderID of given TIPLOC is the max(orderID) for each scheduled service
    # To make this query an edge, specify that tiploc_1's orderID should be 1 more or less than tiploc_2's orderID
    # TODO: For also_passing_tiploc, iterate over result and discard all that don't include it

    c.execute(standard_initial_query)
    for row in c.fetchall():
        past_nodes = []
        # for each potentially matching service
        service = ScheduledService(row[1], row[2], row[3], row[4])
        location_query = f"SELECT tiploc_code, arrival, pass, departure FROM `location` WHERE id = {row[0]} ORDER BY `order` ASC;"
        c.execute(location_query)
        for location in c.fetchall():
            if location[2] is None:
                if location[1] is None:
                    time = location[3]
                else:
                    time = location[1]
            else:
                time = location[2]

            service.add_stop(location[0], time)
            past_nodes.append(location[0])
        if also_passing_tiploc is not None:
            if len(list(set(past_nodes).intersection([tiploc, also_passing_tiploc]))) >= 2:
                services.append(service)
        else:
            services.append(service)
        past_nodes = []
    return services

if __name__ == "__main__":
    test = get_scheduled_services(datetime.date.today(), "EXETRSD", also_passing_tiploc="CWLYBDG")
    print(test)
    print(f"Services found: {len(test)}")

