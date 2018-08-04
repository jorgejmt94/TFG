"""Just provide wrappers to the standard json functions but ones that can
handle dates.
"""
import json
import datetime

import dateutil.parser
import six


class JSONDateEncoder(json.JSONEncoder):

    def default(self, obj):
        if type(obj) == datetime.date:
            # We want to ensure that we have a datetime.
            #
            obj = datetime.datetime.combine(obj, datetime.time.min)

        if isinstance(obj, datetime.datetime):
            return 'datetime:' + obj.isoformat()

        # We don't have a datetime, so we simply fall back to the json
        # encoder.
        #
        return json.JSONEncoder.default(self, obj)


class JSONDateDecoder(json.JSONDecoder):

    def str_to_datetime(self, s):
        try:
            # parse output from datetime.isoformat()
            #
            d = s.replace('datetime:', '')
            return dateutil.parser.parse(d)
        except ValueError:
            # Oops! However it could have been a string that happened to start
            # with 'datetime:'. So return that.
            #
            return s

    def _decode(self, data):
        if isinstance(data, six.string_types) and data.startswith('datetime:'):
            return self.str_to_datetime(data)

        if isinstance(data, list):
            return [self._decode(d) for d in data]

        if isinstance(data, dict):
            d = {}
            for key, value in six.iteritems(data):
                d[key] = self._decode(value)
            return d

        return data

    def decode(self, json_string):
        # First decode it using json.
        #
        decoded_obj = json.JSONDecoder.decode(json.JSONDecoder(), json_string)

        # Now check decoded object for any potential datetime objects within.
        #
        return self._decode(decoded_obj)


def dumps(*args, **kwargs):
    kwargs["cls"] = JSONDateEncoder
    return json.dumps(*args, **kwargs)


def loads(*args, **kwargs):
    kwargs["cls"] = JSONDateDecoder
    return json.loads(*args, **kwargs)


def dump(*args, **kwargs):
    kwargs["cls"] = JSONDateEncoder
    return json.dump(*args, **kwargs)


def load(*args, **kwargs):
    kwargs["cls"] = JSONDateDecoder
    return json.load(*args, **kwargs)


def to_json_serialisable_object(obj):
    """We use this when:
    - we have an object that can't be serialised with json
    - we want the object to be serialised with json!

    Thus we pass it through a jsond and json to get back an object that
    is the same as the original, just with all `datetime` values replaced
    with string values of the form 'datetime: $original_datetime'.

    >>> to_json_serialisable_object({'date': datetime.datetime(2010, 1, 1)})
    {u'date': u'datetime:2010-01-01T00:00:00'}
    """
    obj_as_string = dumps(obj)

    # We want something of the same structure as the original object.
    # So we take the string, and json.loads it to get that back.
    return json.loads(obj_as_string)


def from_json_serialisable_object(obj):
    """We use this when:
    - we have an object that was json serialisable
    - it may have a 'datetime:$date' value
    - we want to convert all those string values back into `datetime` values.

    I.e. this is the opposite of to_json_serialisable_object.
    >>> obj = {u'date': u'datetime:2010-01-01T00:00:00'}
    >>> from_json_serialisable_object(obj)
    {u'date': datetime.datetime(2010, 1, 1, 0, 0)}
    """
    obj_as_string = json.dumps(obj)
    return loads(obj_as_string)


def to_json_serializable_object(obj):
    """Support alternative spellings.
    """
    return to_json_serialisable_object(obj)


def from_json_serializable_object(obj):
    """Support alternative spellings.
    """
    return from_json_serializable_object(obj)
