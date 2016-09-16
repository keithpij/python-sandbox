'''
Google Cloud Storage utilities.
'''
import argparse
import filecmp
import json
import tempfile

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import http


def create_service():
    """Creates the service object for calling the Cloud Storage API."""
    # Get the application default credentials. When running locally, these are
    # available after running `gcloud init`. When running on compute
    # engine, these are available from the environment.
    credentials = GoogleCredentials.get_application_default()

    # Construct the service object for interacting with the Cloud Storage API -
    # the 'storage' service, at version 'v1'.
    # You can browse other available api services and versions here:
    #     https://developers.google.com/api-client-library/python/apis/
    return discovery.build('storage', 'v1', credentials=credentials)


def get_bucket_metadata(bucket):
    """Retrieves metadata about the given bucket."""
    service = create_service()

    # Make a request to buckets.get to retrieve a list of objects in the
    # specified bucket.
    req = service.buckets().get(bucket=bucket)
    return req.execute()


def list_bucket(bucket):
    """Returns a list of metadata of the objects within the given bucket."""
    service = create_service()

    # Create a request to objects.list to retrieve a list of objects.
    fields_to_return = \
        'nextPageToken,items(name,size,contentType,metadata(my-key))'
    req = service.objects().list(bucket=bucket, fields=fields_to_return)

    all_objects = []
    # If you have too many items to list in one request, list_next() will
    # automatically handle paging with the pageToken.
    while req:
        resp = req.execute()
        all_objects.extend(resp.get('items', []))
        req = service.objects().list_next(req, resp)
    return all_objects


def upload_object(bucket, filename, readers, owners):
    service = create_service()

    # This is the request body as specified:
    # http://g.co/cloud/storage/docs/json_api/v1/objects/insert#request
    body = {
        'name': filename,
    }

    # If specified, create the access control objects and add them to the
    # request body
    if readers or owners:
        body['acl'] = []

    for r in readers:
        body['acl'].append({
            'entity': 'user-%s' % r,
            'role': 'READER',
            'email': r
        })
    for o in owners:
        body['acl'].append({
            'entity': 'user-%s' % o,
            'role': 'OWNER',
            'email': o
        })

    # Now insert them into the specified bucket as a media insertion.
    # http://g.co/dev/resources/api-libraries/documentation/storage/v1/python/latest/storage_v1.objects.html#insert
    with open(filename, 'rb') as f:
        req = service.objects().insert(
            bucket=bucket, body=body,
            # You can also just set media_body=filename, but # for the sake of
            # demonstration, pass in the more generic file handle, which could
            # very well be a StringIO or similar.
            media_body=http.MediaIoBaseUpload(f, 'application/octet-stream'))
        resp = req.execute()

    return resp


def get_object(bucket, filename, out_file):
    service = create_service()

    # Use get_media instead of get to get the actual contents of the object.
    # http://g.co/dev/resources/api-libraries/documentation/storage/v1/python/latest/storage_v1.objects.html#get_media
    req = service.objects().get_media(bucket=bucket, object=filename)

    downloader = http.MediaIoBaseDownload(out_file, req)

    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download {}%.".format(int(status.progress() * 100)))

    return out_file


def delete_object(bucket, filename):
    service = create_service()

    req = service.objects().delete(bucket=bucket, object=filename)
    resp = req.execute()

    return resp


if __name__ == '__main__':
    bucket = 'utopian-saga-142816.appspot.com'
    m = get_bucket_metadata(bucket)
    l = list_bucket(bucket)
    print(m)
    print(type(m))

    print(l)
    print(type(l))

    # print(json.dumps(get_bucket_metadata(bucket), indent=2))
    # print(json.dumps(list_bucket(bucket), indent=2))
