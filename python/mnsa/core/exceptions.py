# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import print_function, division, absolute_import


class MnsaError(Exception):
    """A custom core Mnsa exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(MnsaError, self).__init__(message)


class MnsaNotImplemented(MnsaError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(MnsaNotImplemented, self).__init__(message)


class MnsaAPIError(MnsaError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Mnsa API'
        else:
            message = 'Http response error from Mnsa API. {0}'.format(message)

        super(MnsaAPIError, self).__init__(message)


class MnsaApiAuthError(MnsaAPIError):
    """A custom exception for API authentication errors"""
    pass


class MnsaMissingDependency(MnsaError):
    """A custom exception for missing dependencies."""
    pass


class MnsaWarning(Warning):
    """Base warning for Mnsa."""


class MnsaUserWarning(UserWarning, MnsaWarning):
    """The primary warning class."""
    pass


class MnsaSkippedTestWarning(MnsaUserWarning):
    """A warning for when a test is skipped."""
    pass


class MnsaDeprecationWarning(MnsaUserWarning):
    """A warning for deprecated features."""
    pass
