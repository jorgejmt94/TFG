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

import appier

from . import user
from . import drive
from . import token

BASE_URL = "https://www.googleapis.com/"
""" The default base url to be used when no other
base url value is provided to the constructor """

LOGIN_URL = "https://accounts.google.com/o/"
""" Default base url that is going to be used for the
login part of the specification, the oauth login basis """

CLIENT_ID = None
""" The default value to be used for the client id
in case no client id is provided to the api client """

CLIENT_SECRET = None
""" The secret value to be used for situations where
no client secret has been provided to the client """

REDIRECT_URL = "http://localhost:8080/oauth"
""" The redirect url used as default (fallback) value
in case none is provided to the api (client) """

SCOPE = (
    "email",
)
""" The list of permissions to be used to create the
scope string for the oauth value """

class API(
    appier.OAuth2API,
    user.UserAPI,
    drive.DriveAPI,
    token.TokenAPI
):

    def __init__(self, *args, **kwargs):
        appier.OAuth2API.__init__(self, *args, **kwargs)
        self.client_id = appier.conf("GOOGLE_ID", CLIENT_ID)
        self.client_secret = appier.conf("GOOGLE_SECRET", CLIENT_SECRET)
        self.redirect_url = appier.conf("GOOGLE_REDIRECT_URL", REDIRECT_URL)
        self.base_url = kwargs.get("base_url", BASE_URL)
        self.login_url = kwargs.get("login_url", LOGIN_URL)
        self.client_id = kwargs.get("client_id", self.client_id)
        self.client_secret = kwargs.get("client_secret", self.client_secret)
        self.redirect_url = kwargs.get("redirect_url", self.redirect_url)
        self.scope = kwargs.get("scope", SCOPE)
        self.access_token = kwargs.get("access_token", None)
        self.refresh_token = kwargs.get("refresh_token", None)

    def auth_callback(self, params, headers):
        if not self.refresh_token: return
        self.oauth_refresh()
        params["access_token"] = self.get_access_token()
        headers["Authorization"] = "Bearer %s" % self.get_access_token()

    def oauth_authorize(self, state = None, access_type = None, approval_prompt = True):
        url = self.login_url + "oauth2/auth"
        values = dict(
            client_id = self.client_id,
            redirect_uri = self.redirect_url,
            response_type = "code",
            scope = " ".join(self.scope)
        )
        if state: values["state"] = state
        if access_type: values["access_type"] = access_type
        if approval_prompt: values["approval_prompt"] = "force"
        data = appier.legacy.urlencode(values)
        url = url + "?" + data
        return url

    def oauth_access(self, code):
        url = self.login_url + "oauth2/token"
        contents = self.post(
            url,
            token = False,
            client_id = self.client_id,
            client_secret = self.client_secret,
            grant_type = "authorization_code",
            redirect_uri = self.redirect_url,
            code = code
        )
        self.access_token = contents["access_token"]
        self.refresh_token = contents.get("refresh_token", None)
        self.trigger("access_token", self.access_token)
        self.trigger("refresh_token", self.refresh_token)
        return self.access_token

    def oauth_refresh(self):
        url = self.login_url + "oauth2/token"
        contents = self.post(
            url,
            callback = False,
            token = False,
            client_id = self.client_id,
            client_secret = self.client_secret,
            grant_type = "refresh_token",
            redirect_uri = self.redirect_url,
            refresh_token = self.refresh_token
        )
        self.access_token = contents["access_token"]
        self.trigger("access_token", self.access_token)
        return self.access_token
