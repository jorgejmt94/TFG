#!/usr/bin/env python

import urllib
import urllib2
import urlparse
import gzip as gzip_
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
try:
    import simplejson as json
except ImportError:
    import json


API_URL = 'http://translate.googleapis.com/translate_a/t'
#don't bother google with suspicious user-agents
DEFAULT_USER_AGENT = ('Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 '
                      '(KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6')


def translate(text, target_language, source_language=None, html=True,
              user_agent=DEFAULT_USER_AGENT, referer=None, origin=None,
              timeout=None, flat=False, urlopen=urllib2.urlopen, gzip=True):

    if isinstance(text, basestring):
        text = (text,)
    #TODO: split for smaller queries
    #official clients split queries for smaller parts, but service can tranlate
    #blocks for almost 2MB.

    headers = {
        'User-Agent': user_agent,
        'Accept-Charset': 'utf-8',
    }
    if gzip:
        headers['Accept-Encoding'] = 'gzip'
    #better supply information below not to be banned by google
    if referer:
        headers['Referer'] = referer
        headers['Origin'] = '%s://%s/' % urlparse.urlsplit(referer)[:2]
        if origin and origin != headers['Origin']:
            raise ValueError('Origin and referer not matched', origin,
                             headers['Origin'])
    elif origin:
        headers['Origin'] = origin

    qs = (
        #('anno', '3'), #wtf? annotation?
        #If anno=3 and sentences more than 1 it returns original before each
        #sentence. If anno!=3 it returns some meta data as last argument if
        #only one word was specified.
        #NOTE: this isn't happening from official clients for some reason.

        #('client', 'te'), #google javascript translator
        ('client', 'te_lib'), #google-chrome translator
        ('format', html and 'html' or ''), #encodes html-entities in response
        ('v', '1.0'), #google translate version
        ('logld', 'v10'), #obviously protocol version
    )

    data = []
    for block in text:
        if isinstance(block, unicode):
            block = block.encode('utf8')
        else:
            #just to make sure this string is utf8 encoded
            try:
                block.decode('utf8')
            except UnicodeDecodeError:
                raise ValueError('Text must be utf8 or unicode')
        data.append(('q', block))
    data += [
        ('sl', source_language or ''),
        ('tl', target_language),
        ('tc', '1'), #counter of splitted request related to one content
        #('ctt', '1'), #wtf? absent in google-chrome translator
    ]

    req = urllib2.Request(API_URL + '?' + urllib.urlencode(qs),
                          urllib.urlencode(data), headers=headers)
    resp = urlopen(req, **(timeout and {'timeout': timeout} or {}))
    if gzip:
        resp_cont = gzip_.GzipFile(fileobj=StringIO(resp.read()), mode='r').read()
    else:
        resp_cont = resp.read()
    result = json.loads(resp_cont.decode('string-escape'))

    if isinstance(result, list) and isinstance(result[-1], list):
        #excluding word meta, if any (see anno=3 description above).
        result = result[:-1]
    if not source_language:
        #return detected source language as last element if it wasn't specified
        result, source_language = result[:-1], result[-1]
    elif isinstance(result, basestring):
        #this isn't json list, just plain text
        result = (result,)

    if flat:
        return u' '.join(result)
    return tuple(result), source_language


if __name__ == '__main__':
    import optparse, locale
    parser = optparse.OptionParser(usage='usage: %prog -t target_language '
                                   '[-s source_language] text to translate')
    parser.add_option('-t', dest='tl', help='Language of translation')
    parser.add_option('-s', dest='sl', help='Language of source text')
    opts, args = parser.parse_args()
    encoding = locale.getpreferredencoding()
    text = u' '.join([arg.decode(encoding) for arg in args]).strip()

    if not opts.tl or not text:
        parser.error('target language or text to translate not specified')

    translation = translate(text, opts.tl, opts.sl, html=False, flat=True)
    print translation.encode(encoding)
