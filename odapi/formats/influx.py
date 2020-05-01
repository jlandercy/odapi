import sys

from odapi.settings import settings

import pandas as pd

from odapi.interfaces.converter import Converter


class InfluxDB(Converter):

    @staticmethod
    def to_frame(result, timekey='time'):
        frames = []
        for serie in result['series']:
            frame = pd.DataFrame(serie['values'], columns=serie['columns'])
            frame = frame.assign(**serie['tags'])
            frames.append(frame)
        frames = pd.concat(frames)
        frames['time'] = pd.to_datetime(frames['time'])
        frame = frames.rename(columns={'time': timekey})
        return frames

    @staticmethod
    def to_product(frame, measurement='default', timekey='time', tags=None, fields=None):
        tags = tags or ['id']
        fields = fields or ['value']
        points = []
        for row in frame.reset_index().to_dict(orient='records'):
            points.append({
                "measurement": measurement,
                "time": row[timekey].isoformat(),
                "tags": {k: row[k] for k in tags},
                "fields": {k: row[k] for k in fields}
            })
        return points


def main():

    import json
    import influxdb

    from odapi.toolbox.synthetic import FakeSignal
    from odapi import formats

    with (settings.user / "influx.json").open() as fh:
        client = influxdb.InfluxDBClient(**json.load(fh))

    df = FakeSignal.electrical()
    df = df.stack(0).reset_index()
    print(df)

    p = formats.InfluxDB.to_product(df, measurement='DigitalMeter', tags=['meter'], fields=['I (A)', 'U (V)'])
    print(p)

    client.write_points(p)

    t1 = pd.Timestamp.utcnow().floor('1D')
    t0 = t1 - pd.Timedelta('1D')
    q = """
    SELECT * FROM "smartcampus_jupyterhub"."autogen"."DigitalMeter"
    WHERE "time" >= '{}' AND "time" < '{}'
    GROUP BY "meter";
    """.format(t0.isoformat(), t1.isoformat())

    res = client.query(q)
    print(res.raw)

    df = df.set_index('time').sort_values(['time', 'meter'])
    df2 = formats.InfluxDB.to_frame(res.raw)
    df2 = df2.set_index('time').loc[:, df.columns].sort_values(['time', 'meter'])
    print(df2)

    print(df == df2)
    print(df.equals(df2))

    sys.exit(0)


if __name__ == "__main__":
    main()
