"""todo: separate library"""

import os
import posixpath
from urllib.parse import urlparse

sep = "/"
schemesep = "://"
altsep = None
join = posixpath.join
normcase = posixpath.normcase
splitdrive = posixpath.splitdrive
dirname = posixpath.dirname
splitext = posixpath.splitext


def isabs(s: str) -> bool:
    """Test whether a path is absolute"""
    s = os.fspath(s)
    scheme_tail = s.split(schemesep, 1)
    return len(scheme_tail) == 2


def split(p):
    _, bucket, path, _, _, _ = urlparse(p)
    return bucket, path.lstrip("/")


def iss3(s: str) -> bool:
    """Test whether a path is an s3 path"""
    return s.startswith("s3://")
