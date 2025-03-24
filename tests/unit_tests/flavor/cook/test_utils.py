import os
import unittest
from multiprocessing import Value
from unittest.mock import mock_open, patch

# Import the module containing the functions
from flavor.cook.utils import (
    CleanEvent,
    CleanInfoJson,
    GetPaths,
    SaveInfoJson,
    SetEvent,
    SetPaths,
    WaitEvent,
    compareVersion,
)


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions in flavor.cook.utils module."""

    def setUp(self):
        """Set up common test variables."""
        self.output_path = "/tmp"
        self.model_path = "/tmp/model.pt"
        self.test_event = "TrainStarted"
        self.test_info = {"key": "value"}

    # ========== SetEvent Tests ==========

    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    @patch("builtins.open", new_callable=mock_open)
    def test_SetEvent_creates_event_file_when_path_exists(self, mock_file):
        """Test that SetEvent creates an event file when OUTPUT_PATH exists."""
        SetEvent(self.test_event)
        mock_file.assert_called_with(f"{self.output_path}/{self.test_event}", "w")

    @patch("os.environ.get")
    def test_SetEvent_raises_error_when_output_path_missing(self, mock_env_get):
        """Test that SetEvent raises EnvironmentError when OUTPUT_PATH is not set."""
        # Configure the mock to return None when os.environ.get('OUTPUT_PATH') is called
        mock_env_get.side_effect = (
            lambda key, default=None: None if key == "OUTPUT_PATH" else default
        )

        with self.assertRaises(EnvironmentError) as context:
            SetEvent(self.test_event)
        self.assertEqual(str(context.exception), "OUTPUT_PATH environment variable is not set.")

    def test_SetEvent_raises_error_for_unknown_event(self):
        """Test that SetEvent raises ValueError for unknown event types."""
        with self.assertRaises(ValueError):
            SetEvent("UnknownEvent")

    # ========== CleanEvent Tests ==========

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_CleanEvent_removes_existing_file(self, mock_remove, mock_exists):
        """Test that CleanEvent removes the event file when it exists."""
        CleanEvent(self.test_event)
        mock_remove.assert_called_with(f"{self.output_path}/{self.test_event}")

    @patch("os.path.exists", return_value=False)
    @patch("os.remove")
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_CleanEvent_does_nothing_when_file_not_exists(self, mock_remove, mock_exists):
        """Test that CleanEvent doesn't try to remove nonexistent event files."""
        CleanEvent(self.test_event)
        mock_remove.assert_not_called()

    # ========== WaitEvent Tests ==========

    @patch("os.path.exists", side_effect=[False, False, True])
    @patch("os.remove")
    @patch("time.sleep", return_value=None)
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_WaitEvent_polls_until_file_exists(self, mock_sleep, mock_remove, mock_exists):
        """Test that WaitEvent polls until the event file exists."""
        is_error = Value("i", 0)
        WaitEvent(self.test_event, is_error)

        # Verify we checked for file existence 3 times
        self.assertEqual(mock_exists.call_count, 3)

        # Verify we removed the file once detected
        mock_remove.assert_called_once_with(os.path.join(self.output_path, self.test_event))

        # Verify we slept twice (before the file was detected)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("os.path.exists", return_value=False)
    @patch("os.remove")
    @patch("time.sleep", return_value=None)
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_WaitEvent_stops_polling_when_error_detected(
        self, mock_sleep, mock_remove, mock_exists
    ):
        """Test that WaitEvent stops polling when an error is detected."""
        is_error = Value("i", 1)
        WaitEvent(self.test_event, is_error)

        # Verify we only checked for file existence once
        self.assertEqual(mock_exists.call_count, 1)

        # Verify we didn't try to remove anything
        mock_remove.assert_not_called()

        # Verify we slept once
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("os.environ.get")
    def test_WaitEvent_raises_error_when_output_path_missing(self, mock_env_get):
        """Test that WaitEvent raises EnvironmentError when OUTPUT_PATH is not set."""
        # Configure the mock to return None when os.environ.get('OUTPUT_PATH') is called
        mock_env_get.side_effect = (
            lambda key, default=None: None if key == "OUTPUT_PATH" else default
        )

        with self.assertRaises(EnvironmentError) as context:
            WaitEvent(self.test_event)
        self.assertEqual(str(context.exception), "OUTPUT_PATH environment variable is not set.")

    # ========== SaveInfoJson Tests ==========

    @patch("builtins.open", new_callable=mock_open)
    @patch.dict(os.environ, {"LOCAL_MODEL_PATH": "/tmp/model.pt"})
    def test_SaveInfoJson_writes_json_to_correct_location(self, mock_file):
        """Test that SaveInfoJson writes JSON to the correct location."""
        SaveInfoJson(self.test_info)
        mock_file.assert_called_with("/tmp/info.json", "w")

    @patch("os.environ.get")
    def test_SaveInfoJson_raises_error_when_model_path_missing(self, mock_env_get):
        """Test that SaveInfoJson raises EnvironmentError when LOCAL_MODEL_PATH is not set."""
        # Configure the mock to return None when os.environ.get('LOCAL_MODEL_PATH') is called
        mock_env_get.side_effect = (
            lambda key, default=None: None if key == "LOCAL_MODEL_PATH" else default
        )

        with self.assertRaises(EnvironmentError) as context:
            SaveInfoJson(self.test_info)
        self.assertEqual(
            str(context.exception), "LOCAL_MODEL_PATH environment variable is not set."
        )

    # ========== CleanInfoJson Tests ==========

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch.dict(os.environ, {"LOCAL_MODEL_PATH": "/tmp/model.pt"})
    def test_CleanInfoJson_removes_existing_file(self, mock_remove, mock_exists):
        """Test that CleanInfoJson removes the info.json file when it exists."""
        CleanInfoJson()
        mock_remove.assert_called_with("/tmp/info.json")

    @patch("os.path.exists", return_value=False)
    @patch("os.remove")
    @patch.dict(os.environ, {"LOCAL_MODEL_PATH": "/tmp/model.pt"})
    def test_CleanInfoJson_does_nothing_when_file_not_exists(self, mock_remove, mock_exists):
        """Test that CleanInfoJson doesn't try to remove nonexistent info.json."""
        CleanInfoJson()
        mock_remove.assert_not_called()

    # ========== SetPaths Tests ==========

    @patch("builtins.open", new_callable=mock_open)
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_SetPaths_handles_single_string_input(self, mock_file):
        """Test that SetPaths correctly handles a single string input."""
        SetPaths("localModels", "model1.pt")
        mock_file.assert_called_with(f"{self.output_path}/localModels", "w")
        mock_file().write.assert_called_with("model1.pt")

    @patch("builtins.open", new_callable=mock_open)
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_SetPaths_handles_list_input(self, mock_file):
        """Test that SetPaths correctly handles a list of paths."""
        SetPaths("localModels", ["model1.pt", "model2.pt"])
        mock_file.assert_called_with(f"{self.output_path}/localModels", "w")
        mock_file().write.assert_called_with("model1.pt\nmodel2.pt")

    def test_SetPaths_raises_error_for_unknown_filename(self):
        """Test that SetPaths raises ValueError for unknown filenames."""
        with self.assertRaises(ValueError):
            SetPaths("unknownFile", "data")

    # ========== GetPaths Tests ==========

    @patch("builtins.open", new_callable=mock_open, read_data="model1.pt\nmodel2.pt")
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_GetPaths_returns_correct_data(self, mock_file):
        """Test that GetPaths returns the correct data from a file."""
        result = GetPaths("localModels")
        self.assertEqual(result, ["model1.pt", "model2.pt"])

    @patch("os.environ.get")
    def test_GetPaths_raises_error_when_output_path_missing(self, mock_env_get):
        """Test that GetPaths raises EnvironmentError when OUTPUT_PATH is not set."""
        # Configure the mock to return None when os.environ.get('OUTPUT_PATH') is called
        mock_env_get.side_effect = (
            lambda key, default=None: None if key == "OUTPUT_PATH" else default
        )

        with self.assertRaises(EnvironmentError) as context:
            GetPaths("localModels")
        self.assertEqual(str(context.exception), "OUTPUT_PATH environment variable is not set.")

    def test_GetPaths_raises_error_for_unknown_filename(self):
        """Test that GetPaths raises ValueError for unknown filenames."""
        with self.assertRaises(ValueError):
            GetPaths("unknownFile")

    # ========== compareVersion Tests ==========

    def test_compareVersion_identifies_equal_versions(self):
        """Test that compareVersion returns 0 for equal versions."""
        result = compareVersion("1.0.0", "1.0.0")
        self.assertEqual(result, 0)

    def test_compareVersion_identifies_lesser_version(self):
        """Test that compareVersion returns -1 when v1 is less than v2."""
        result = compareVersion("1.0.0", "1.0.1")
        self.assertEqual(result, -1)

    def test_compareVersion_identifies_greater_version(self):
        """Test that compareVersion returns 1 when v1 is greater than v2."""
        result = compareVersion("1.0.1", "1.0.0")
        self.assertEqual(result, 1)

    def test_compareVersion_raises_error_for_non_integer_version(self):
        """Test that compareVersion raises ValueError for non-integer version components."""
        with self.assertRaises(ValueError):
            compareVersion("1.0a", "1.0.0")


if __name__ == "__main__":
    unittest.main()
