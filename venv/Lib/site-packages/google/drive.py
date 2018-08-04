#!/usr/bin/python
# -*- coding: utf-8 -*-

# Hive Google API
# Copyright (c) 2008-2017 Hive Solutions Lda.
#
# This file is part of Hive Google API.
#
# Hive Google API is free software: you can redistribute it and/or modify
# it under the terms of the Apache License as published by the Apache
# Foundation, either version 2.0 of the License, or (at your option) any
# later version.
#
# Hive Google API is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details.
#
# You should have received a copy of the Apache License along with
# Hive Google API. If not, see <http://www.apache.org/licenses/>.

__author__ = "João Magalhães <joamag@hive.pt>"
""" The author(s) of the module """

__version__ = "1.0.0"
""" The version of the module """

__revision__ = "$LastChangedRevision$"
""" The revision number of the module """

__date__ = "$LastChangedDate$"
""" The last change date of the module """

__copyright__ = "Copyright (c) 2008-2017 Hive Solutions Lda."
""" The copyright for the module """

__license__ = "Apache License, Version 2.0"
""" The license for the module """

import json

import appier

class DriveAPI(object):

    def list_drive(self, query = None):
        url = self.base_url + "drive/v2/files"
        contents = self.get(url, q = query)
        return contents["items"]

    def insert_drive(
        self,
        data,
        content_type = "application/octet-stream",
        title = None,
        parents = None
    ):
        data = appier.legacy.bytes(data)
        metadata = dict()
        if title: metadata["title"] = title
        if parents: metadata["parents"] = parents
        metadata_s = json.dumps(metadata)
        is_unicode = appier.legacy.is_unicode(metadata_s)
        if is_unicode: metadata_s = metadata_s.encode("utf-8")
        metadata_p = {
            "Content-Type" : "application/json;charset=utf-8",
            "data" : metadata_s
        }
        media_p = {
            "Content-Type" : content_type,
            "data" : data
        }
        url = self.base_url + "upload/drive/v2/files"
        contents = self.post(
            url,
            params = dict(
                uploadType = "multipart"
            ),
            data_m = dict(file = [metadata_p, media_p]),
            mime = "multipart/related"
        )
        return contents

    def folder_drive(self, title, parent = "root", overwrite = False):
        query = "title = '%s' and '%s' in parents and trashed = false" % (title, parent)
        contents = self.list_drive(query = query)
        if contents:
            previous = contents[0]
            if overwrite: self.delete_drive(previous["id"])
            else: return previous
        metadata = dict(
            title = title,
            parents = [dict(id = parent)],
            mimeType = "application/vnd.google-apps.folder"
        )
        metadata_s = json.dumps(metadata)
        is_unicode = appier.legacy.is_unicode(metadata_s)
        if is_unicode: metadata_s = metadata_s.encode("utf-8")
        metadata_p = {
            "Content-Type" : "application/json;charset=utf-8",
            "data" : metadata_s
        }
        url = self.base_url + "upload/drive/v2/files"
        contents = self.post(
            url,
            params = dict(
                uploadType = "multipart"
            ),
            data_m = dict(file = [metadata_p]),
            mime = "multipart/related"
        )
        return contents

    def get_drive(self, id):
        url = self.base_url + "drive/v2/files/%s" % id
        contents = self.get(url)
        return contents

    def delete_drive(self, id):
        url = self.base_url + "drive/v2/files/%s" % id
        contents = self.delete(url)
        return contents

    def children_drive(self, id = "root"):
        url = self.base_url + "drive/v2/files/%s/children" % id
        contents = self.get(url)
        return contents

    def remove_drive(self, title, parent = "root"):
        query = "title = '%s' and '%s' in parents and trashed = false" % (title, parent)
        contents = self.list_drive(query = query)
        if not contents: return
        previous = contents[0]
        self.delete_drive(previous["id"])
