"""Wrapper around Google Drive API (v3) to download files from GDrive"""

import io
import os.path
import pickle
import shutil
from mimetypes import MimeTypes
from typing import List, Optional

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from tqdm import tqdm

from src.constants import PROJECT_PATH

SECRETS_PATH = PROJECT_PATH / "secrets"


class DriveAPI:
    """
    Python wrapper around the google drive v3 API.

    Handels OAuth connection, file browsing, meta data retrieval, file download and
    file upload.
    """

    # Define the scopes
    SCOPES = [  #'https://www.googleapis.com/auth/drive',  # For reading and writing
        "https://www.googleapis.com/auth/drive.readonly"  # For reading only
    ]
    # Define GDrive types
    GDRIVE_FOLDER = "application/vnd.google-apps.folder"

    def __init__(self, credentials_path: str = SECRETS_PATH / "credentials.json"):

        # Variable self.creds will store the user access token.
        # If no valid token found we will create one.
        self.creds = None
        self.credentials_path = credentials_path
        self._user_data = None

        # Authenticate
        self._authenticate()

        # Connect to the API service
        self.service = build("drive", "v3", credentials=self.creds)

    def _authenticate(self) -> None:
        """
        Authenticate user with user token from google OAuth 2.0.
        """
        # The file token.pickle stores the user's access and refresh tokens. It is
        # created automatically when the authorization flow completes for the first
        # time.

        # Check if file token.pickle exists
        if os.path.exists(SECRETS_PATH / "token.pickle"):

            # Read the token from the file and
            # store it in the variable self.creds
            with open(SECRETS_PATH / "token.pickle", "rb") as token:
                self.creds = pickle.load(token)

        # If no valid credentials are available,
        # request the user to log in.
        if not self.creds or not self.creds.valid:

            # If token is expired, it will be refreshed,
            # else, we will request a new one.
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                self._perform_oauth()

            # Save the access token in token.pickle
            # file for future usage
            with open(SECRETS_PATH / "token.pickle", "wb") as token:
                pickle.dump(self.creds, token)

    def _perform_oauth(self) -> None:
        """
        Perform google OAuth 2.0 flow to authenticate user.
        """
        flow = InstalledAppFlow.from_client_secrets_file(
            self.credentials_path, DriveAPI.SCOPES
        )
        self.creds = flow.run_local_server(port=0)

    @property
    def user_data(self) -> dict:
        """Returns metadata of currently logged in user"""
        if self._user_data is None:
            # fetch user data
            about = self.service.about()
            self._user_data = about.get(fields="user").execute()["user"]
        return self._user_data

    @property
    def user_email(self) -> str:
        """Returns email address of currently logged in user"""
        return self.user_data["emailAddress"]

    @property
    def username(self) -> str:
        """Returns user name of currently logged in user"""
        return self.user_data["displayName"]

    def file_download(
        self, file_id: str, save_path: str, chunksize: int = 200 * 1024 * 1024
    ) -> bool:
        """
        Download file with given `file_id` and save in `save_path`.

        Raises an error if the download fails.

        Args:
            file_id (str): id of the file to download
            save_path (str): path where the file will be saved
            chunksize (int, optional): size of the chunks of data to request with
                each http request. If the download is slow, try increasing the chunksize
                as google limits the number of http requests we can pose per second.
                Defaults to 200*1024*1024 (= 200 MB).

        Returns:
            bool: True, iff the file was downloaded successfully.
        """
        request = self.service.files().get_media(fileId=file_id)
        file_handle = io.BytesIO()

        # Initialise a downloader object to download the file
        downloader = MediaIoBaseDownload(file_handle, request, chunksize=chunksize)
        done = False

        print("Starting file download")
        progress_bar = tqdm(total=100)
        while not done:
            status, done = downloader.next_chunk()
            if status:
                progress_bar.update(n=status.progress() * 100)
        progress_bar.close()

        file_handle.seek(0)

        # Write the received data to the file
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file_handle, f)

        print("File Downloaded")
        # Return True if file Downloaded successfully
        return True

    def file_upload(self, file_path: str, save_name: Optional[str] = None) -> bool:
        """
        Uploads file at `file_path` to gdrive.

        Raises an error if upload fails. Filename is taken from the filename

        Args:
            file_path (str): path of the file to upload
            save_name (Optional[str]): name with which to save the file.
                Defaults to None.

        Returns:
            bool: True, iff upload succeeded.
        """

        if not save_name:
            # Extract the file name out of the file path
            save_name = file_path.split("/")[-1]

        # Find the MimeType of the file
        mimetype = MimeTypes().guess_type(save_name)[0]

        # create file metadata
        file_metadata = {"name": save_name}

        media = MediaFileUpload(file_path, mimetype=mimetype)

        # Create a new file in the Drive storage
        file_creation = self.service.files().create(
            body=file_metadata, media_body=media, fields="id"
        )
        file_creation.execute()

        print("File Uploaded.")
        return True

    def get_mimetype(self, file_id: str) -> str:
        """
        Returns mime type of the given file

        Args:
            file_id (str): id of the file

        Returns:
            str: mime type of the given file
        """

        query = self.service.files().get(fileId=file_id, fields="mimeType")
        mime_type = query.execute()["mimeType"]

        return mime_type

    def is_mimetype(self, file_id: str, target_mime_type: str) -> bool:
        """
        Check mime type of a given file against target mime type

        Args:
            file_id (str): id of the file
            target_mime_type (str): target mime type to check against

        Returns:
            bool: True, iff the mime type of the given file matches the target mime
                type
        """

        return self.get_mimetype(file_id) == target_mime_type

    def is_folder(self, file_id: str) -> bool:
        """
        Checks if a given file is a gdrive folder

        Args:
            file_id (str): id of the file

        Returns:
            bool: True, iff file is a gdrive folder
        """

        return self.is_mimetype(
            file_id=file_id, target_mime_type=DriveAPI.GDRIVE_FOLDER
        )

    def is_tif(self, file_id: str) -> bool:
        """
        Checks if a given file is a .tiff file.

        Args:
            file_id (str): id of the file

        Returns:
            bool: True, iff file is of type .tiff
        """

        return self.is_mimetype(file_id, target_mime_type="image/tiff")

    def get_folder(self, folder_name: str) -> dict:
        """
        Return metadata of gdrive folder with the given `folder_name`

        Raises an error if `folder_name` does not identify a unique folder (or does
        not exist).

        Args:
            folder_name (str): The name of the folder for which to obtain metadata

        Returns:
            dict: The metadata of the requested folder.
        """

        file_browser = self.service.files()
        query = file_browser.list(
            q=f"name='{folder_name}' and mimeType='{self.GDRIVE_FOLDER}'"
        )
        result = query.execute()["files"]
        assert len(result) == 1, "None or multiple folders with this name exist."

        return result[0]

    def get_folder_id(self, folder_name: str) -> str:
        """
        Return id of a folder with the given name.

        Raises an error if folder does not exist or if multiple folder share the
        same name.

        Args:
            folder_name (str): The folder whose id should be returned

        Returns:
            str: id of the folder with the given foldername.
        """

        folder = self.get_folder(folder_name)

        return folder["id"]

    def get_file_name(self, file_id: str) -> str:
        """
        Get name of a file by id

        Args:
            file_id (str): The id of the file whose name should be returned

        Returns:
            str: The filename
        """

        query = self.service.files().get(fileId=file_id, fields="name")
        return query.execute()["name"]

    def list_all_files(self) -> List[dict]:
        """
        Lists all files which are not folders in gdrive

        Returns:
            List[dict]: A list of all files in the given gdrive account
        """

        file_browser = self.service.files()
        query = file_browser.list(q=f"mimeType!='{self.GDRIVE_FOLDER}'")
        return query.execute()["files"]

    def list_all_folders(self) -> List[dict]:
        """
        List all folders in gdrive

        Returns:
            List[dict]: List of all folders and their id's
        """

        file_browser = self.service.files()
        query = file_browser.list(q=f"mimeType='{self.GDRIVE_FOLDER}'")
        return query.execute()["files"]

    def list_files_in_folder(
        self, folder_id: str, fields: str = "files (id, name)", **kwargs
    ) -> List[dict]:
        """
        List all files in a gdrive folder with given `folder_id`.

        Args:
            folder_id (str): The id of the folder
            fields (str, optional): The fields to list. Possible values can be taken
            from the gdrive api v3 documentation. Defaults to "files (id, name)".

        Returns:
            List[dict]: A list of all the files in the given folder.
        """

        file_browser = self.service.files()

        assert self.is_folder(folder_id), "Selected file is not a folder"

        query = file_browser.list(
            q=f"'{folder_id}' in parents", fields=fields, **kwargs
        )
        return query.execute()["files"]
