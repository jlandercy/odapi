import sys
import json

import pandas as pd
import influxdb

from odapi.settings import settings
from odapi.formats import InfluxDB
from odapi import connectors


def feed(connection, **kwargs):

    # Get data:
    client = getattr(connectors, kwargs.get("connector", "Irceline"))()
    meta = client.select(**kwargs.get("channels", {})).loc[:,kwargs.get("tagkeys", slice(None))]
    recs = client.get_records(meta, **kwargs.get("records", {})).dropna(subset=kwargs.get("dropna", []))
    data = recs.merge(meta).assign(**kwargs.get('tags', {}))
    tagkeys = set(kwargs.get("tagkeys", {})).union(set(kwargs.get('tags', {}).keys()))

    # Create Points:
    points = InfluxDB.to_product(data, timekey=kwargs.get("timekey", "timestamp"),
                                 measurement=kwargs.get("measurement", "default"),
                                 fields=tuple(kwargs.get("fields", [])),
                                 tags=tuple(tagkeys))
    # Write Points:
    if points:
        connection.write_points(points)
        settings.logger.info("INFLUX sent {} point(s)".format(len(points)))


def main():

    import argparse
    import pathlib

    from influxdb import InfluxDBClient
    #from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.schedulers.background import BackgroundScheduler

    # CLI Arguments:
    clargs = argparse.ArgumentParser(description='Irceline Service')
    clargs.add_argument('--program', type=str, default=str(settings.user / 'program.json'),
                        help='Program definition path (eg.: ~/odapi/program.json)')
    args = clargs.parse_args()

    # Create Scheduler:
    sched = BackgroundScheduler(timezone='UTC')
    t1 = pd.Timestamp.utcnow()

    # Read Program file:
    with pathlib.Path(args.program).open() as fh:
        prog = json.load(fh)
    settings.logger.info("SCHED Program loaded with {} job(s)".format(len(prog)))

    # Add Jobs:
    for k, v in prog.items():

        dbcon = InfluxDBClient(**v["credentials"])
        settings.logger.info("INFLUX created client: {}".format(dbcon))
        settings.logger.info("INFLUX allowed databases: {}".format(dbcon.get_list_database()))

        t = t1.floor('1S').replace(**v.get('sync', {}))
        sched.add_job(feed, 'interval', id=k, **v['interval'], next_run_time=t1,
                      misfire_grace_time=5, coalesce=True, args=(dbcon,), kwargs=v)
        settings.logger.info("SCHED job '{}' registred (sync={}): {}".format(k, t, v))

    # Start Service:
    settings.logger.info("SCHED Service started")
    try:
        sched.start()
    except KeyboardInterrupt:
        sched.shutdown()
        settings.logger.warning("SCHED Service stopped")

    sys.exit(0)


if __name__ == "__main__":
    main()
