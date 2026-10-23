import json
import os
from datetime import datetime
from typing import Dict, List, Union, Optional
import uuid


class TimeTracker:
    "Track elapsed time in a JSON file. Preserve existing records across initialization, append records with computed durations, and support repeated reading and querying."

    def __init__(self, file_path: str, filename: str = "time_records.json"):
        "Initialize an incremental time tracker at file_path with filename, defaulting to time_records.json."
        self.file_path = file_path
        self.filename = filename
        self.full_path = os.path.join(file_path, filename)

        # Ensure the directory exists.
        os.makedirs(file_path, exist_ok=True)

        # Initialize or load the JSON file incrementally.
        self._initialize_or_load_json_file()

        print(f"""Time tracker initialized: {self.full_path}""")

    def _initialize_or_load_json_file(self) -> None:
        "Initialize or load a JSON file while preserving existing data."
        if not os.path.exists(self.full_path):
            # Create the file if it does not exist.
            initial_data = {}
            self._save_json_data(initial_data)
            print(f"""Created time log: {self.full_path}""")
        else:
            data = self._read_json_data()
            # Update the last-access time.
            print(f"""Loaded existing time log: {self.full_path}""")

    def _parse_time(self, time_input: Union[str, datetime]) -> datetime:
        "Parse time_input, a string or datetime object, using supported formats. Return a datetime."
        if isinstance(time_input, datetime):
            return time_input

        # Supported time formats
        time_formats = [
            "%Y-%m-%d %H:%M:%S",  # 2024-01-01 10:30:00
            "%Y-%m-%d %H:%M",  # 2024-01-01 10:30
            "%m-%d %H:%M",  # 01-01 10:30
            "%H:%M:%S",  # 10:30:00
            "%H:%M"  # 10:30
        ]

        for fmt in time_formats:
            try:
                if fmt in ["%H:%M:%S", "%H:%M"]:
                    # Use today's date for time-only inputs.
                    time_obj = datetime.strptime(time_input, fmt)
                    return datetime.now().replace(
                        hour=time_obj.hour,
                        minute=time_obj.minute,
                        second=time_obj.second if fmt == "%H:%M:%S" else 0,
                        microsecond=0
                    )
                elif fmt == "%m-%d %H:%M":
                    # Use the current year when the year is absent.
                    time_obj = datetime.strptime(f"{datetime.now().year}-{time_input}", "%Y-%m-%d %H:%M")
                    return time_obj
                else:
                    return datetime.strptime(time_input, fmt)
            except ValueError:
                continue

        raise ValueError(f"""Unable to parse time format: {time_input}""")


    def add_record(self, start_time,
                   end_time,
                   record_name: str,
                   time_reset: bool=True) -> Dict:
        "Append a record with start_time, end_time and record_name without changing existing records. Return the new record dictionary."
        try:
            # Parse timestamps.


            # Validate temporal order.
            if end_time <= start_time:
                raise ValueError("End time must be later than start time")

            # Calculate elapsed time in minutes.
            duration_minutes = round((end_time - start_time)/ 60, 4)

            # Read current data to include records added by other instances.
            data = self._read_json_data()

            if time_reset:
                data[record_name] = duration_minutes
            else:
                data[record_name] += duration_minutes
            # Save data.
            self._save_json_data(data)

            print(f"""Added record: {record_name} (duration: {duration_minutes} minutes)""")

        except Exception as e:
            print(f"""Failed to add record: {str(e)}""")
            raise

    def _read_json_data(self) -> Dict:
        "Read the latest JSON data from disk on every call."
        try:
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Reinitialize if the file was deleted.
            print("Warning: JSON file was deleted; recreating")
            self._initialize_or_load_json_file()
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"""Failed to read JSON file: {str(e)}""")
            raise

    def _save_json_data(self, data: Dict) -> None:
        "Save data to a JSON file."
        try:
            # Ensure the directory exists.
            os.makedirs(self.file_path, exist_ok=True)

            # Write atomically through a temporary file.
            temp_path = f"{self.full_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomically replace the destination.
            os.replace(temp_path, self.full_path)

        except Exception as e:
            # Clean up the temporary file.
            temp_path = f"{self.full_path}.tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"""Failed to save JSON file: {str(e)}""")
            raise



class ThinkTracker:
    "Track elapsed time in a JSON file. Preserve existing records across initialization, append records with computed durations, and support repeated reading and querying."

    def __init__(self, file_path: str, filename: str = "think_records.json"):
        "Initialize an incremental reasoning tracker at file_path with filename, defaulting to think_records.json."
        self.file_path = file_path
        self.filename = filename
        self.full_path = os.path.join(file_path, filename)

        # Ensure the directory exists.
        os.makedirs(file_path, exist_ok=True)

        # Initialize or load the JSON file incrementally.
        self._initialize_or_load_json_file()

        print(f"""Time tracker initialized: {self.full_path}""")

    def _initialize_or_load_json_file(self) -> None:
        "Initialize or load a JSON file while preserving existing data."
        if not os.path.exists(self.full_path):
            # Create the file if it does not exist.
            initial_data = {}
            self._save_json_data(initial_data)
            print(f"""Created reasoning log: {self.full_path}""")
        else:
            data = self._read_json_data()
            # Update the last-access time.
            print(f"""Loaded existing reasoning log: {self.full_path}""")

    def _parse_time(self, time_input: Union[str, datetime]) -> datetime:
        "Parse time_input, a string or datetime object, using supported formats. Return a datetime."
        if isinstance(time_input, datetime):
            return time_input

        # Supported time formats
        time_formats = [
            "%Y-%m-%d %H:%M:%S",  # 2024-01-01 10:30:00
            "%Y-%m-%d %H:%M",  # 2024-01-01 10:30
            "%m-%d %H:%M",  # 01-01 10:30
            "%H:%M:%S",  # 10:30:00
            "%H:%M"  # 10:30
        ]

        for fmt in time_formats:
            try:
                if fmt in ["%H:%M:%S", "%H:%M"]:
                    # Use today's date for time-only inputs.
                    time_obj = datetime.strptime(time_input, fmt)
                    return datetime.now().replace(
                        hour=time_obj.hour,
                        minute=time_obj.minute,
                        second=time_obj.second if fmt == "%H:%M:%S" else 0,
                        microsecond=0
                    )
                elif fmt == "%m-%d %H:%M":
                    # Use the current year when the year is absent.
                    time_obj = datetime.strptime(f"{datetime.now().year}-{time_input}", "%Y-%m-%d %H:%M")
                    return time_obj
                else:
                    return datetime.strptime(time_input, fmt)
            except ValueError:
                continue

        raise ValueError(f"""Unable to parse time format: {time_input}""")


    def add_record(self,
                   module: str,
                   think,
                   sub_module=None) -> Dict:
        "Append a record without changing existing entries."
        try:
            # Parse timestamps.


            # Validate temporal order.

            # Read current data to include records added by other instances.
            data = self._read_json_data()
            if sub_module is None:
                data[module] = think

            elif data.get(module, -1) == -1 and sub_module is not None:
                data[module] = {}
                data[module][sub_module] = think

            elif data.get(module, -1) != -1 and sub_module is not None:

                if isinstance(data[module], dict):
                    data[module][sub_module] = think
                else:
                    data[module] = {}
                    data[module][sub_module] = think

            # Save data.
            self._save_json_data(data)

            print(f"""Added reasoning for {module} ({sub_module}): {think}""")

        except Exception as e:
            print(f"""Failed to add record: {str(e)}""")
            raise

    def _read_json_data(self) -> Dict:
        "Read the latest JSON data from disk on every call."
        try:
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Reinitialize if the file was deleted.
            print("Warning: JSON file was deleted; recreating")
            self._initialize_or_load_json_file()
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"""Failed to read JSON file: {str(e)}""")
            raise

    def _save_json_data(self, data: Dict) -> None:
        "Save data to a JSON file."
        try:
            # Ensure the directory exists.
            os.makedirs(self.file_path, exist_ok=True)

            # Write atomically through a temporary file.
            temp_path = f"{self.full_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomically replace the destination.
            os.replace(temp_path, self.full_path)

        except Exception as e:
            # Clean up the temporary file.
            temp_path = f"{self.full_path}.tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"""Failed to save JSON file: {str(e)}""")
            raise


class TokenTracker:
    "Track elapsed time in a JSON file. Preserve existing records across initialization, append records with computed durations, and support repeated reading and querying."

    def __init__(self, file_path: str, filename: str = "think_records.json", module=None, reset=True):
        "Initialize an incremental reasoning tracker at file_path with filename, defaulting to think_records.json."
        self.file_path = file_path
        self.filename = filename
        self.full_path = os.path.join(file_path, filename)
        self.module = module
        self.reset = reset

        # Ensure the directory exists.
        os.makedirs(file_path, exist_ok=True)

        # Initialize or load the JSON file incrementally.
        self._initialize_or_load_json_file()

        print(f"""Token tracker initialized: {self.full_path}""")

    def _initialize_or_load_json_file(self) -> None:
        "Initialize or load a JSON file while preserving existing data."
        if not os.path.exists(self.full_path):
            # Create the file if it does not exist.
            initial_data = {self.module: {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
            self._save_json_data(initial_data)
            print(f"""Created token log: {self.full_path}""")
            print(f"""Current token counts for {self.module}: prompt_tokens: {initial_data[self.module]['prompt_tokens']}, """
                  f"completion_tokens: {initial_data[self.module]['completion_tokens']}, "
                  f"total_tokens: {initial_data[self.module]['total_tokens']}")
        else:
            data = self._read_json_data()
            if self.reset:
                data[self.module] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            self._save_json_data(data)
            # Update the last-access time.
            print(f"""Loaded existing token log: {self.full_path}""")
            print(f"""Loaded token counts for {self.module}: prompt_tokens: {data[self.module]['prompt_tokens']}, """
                  f"completion_tokens: {data[self.module]['completion_tokens']}, "
                  f"total_tokens: {data[self.module]['total_tokens']}")


    def _parse_time(self, time_input: Union[str, datetime]) -> datetime:
        "Parse time_input, a string or datetime object, using supported formats. Return a datetime."
        if isinstance(time_input, datetime):
            return time_input

        # Supported time formats
        time_formats = [
            "%Y-%m-%d %H:%M:%S",  # 2024-01-01 10:30:00
            "%Y-%m-%d %H:%M",  # 2024-01-01 10:30
            "%m-%d %H:%M",  # 01-01 10:30
            "%H:%M:%S",  # 10:30:00
            "%H:%M"  # 10:30
        ]

        for fmt in time_formats:
            try:
                if fmt in ["%H:%M:%S", "%H:%M"]:
                    # Use today's date for time-only inputs.
                    time_obj = datetime.strptime(time_input, fmt)
                    return datetime.now().replace(
                        hour=time_obj.hour,
                        minute=time_obj.minute,
                        second=time_obj.second if fmt == "%H:%M:%S" else 0,
                        microsecond=0
                    )
                elif fmt == "%m-%d %H:%M":
                    # Use the current year when the year is absent.
                    time_obj = datetime.strptime(f"{datetime.now().year}-{time_input}", "%Y-%m-%d %H:%M")
                    return time_obj
                else:
                    return datetime.strptime(time_input, fmt)
            except ValueError:
                continue

        raise ValueError(f"""Unable to parse time format: {time_input}""")


    def add_record(self,
                   module: str=None,
                   prompt_tokens=0,
                   completion_tokens=0,
                   total_tokens=0) -> Dict:
        "Append a record without changing existing entries."
        try:
            # Parse timestamps.


            # Validate temporal order.

            # Read current data to include records added by other instances.
            data = self._read_json_data()

            if module is None:
                module = self.module

            if data.get(module, -1) == -1:
                data[module] = {}

            if data[module].get('prompt_tokens', -1) == -1:
                data[module]['prompt_tokens'] = 0

            if data[module].get('completion_tokens', -1) == -1:
                data[module]['completion_tokens'] = 0

            if data[module].get('total_tokens', -1) == -1:
                data[module]['total_tokens'] = 0

            data[module]['prompt_tokens'] = data[module]['prompt_tokens'] + prompt_tokens
            data[module]['completion_tokens'] = data[module]['completion_tokens'] + completion_tokens
            data[module]['total_tokens'] = data[module]['total_tokens'] + total_tokens

            # Save data.
            self._save_json_data(data)

            print(f"""Added tokens / cumulative tokens for {module}: prompt_tokens: {prompt_tokens}/{data[module]['prompt_tokens']}, """
                  f"completion_tokens: {completion_tokens}/{data[module]['completion_tokens']}, "
                  f"total_tokens: total_tokens: {total_tokens}/{data[module]['total_tokens']}")

        except Exception as e:
            print(f"""Failed to add record: {str(e)}""")
            raise

    def _read_json_data(self) -> Dict:
        "Read the latest JSON data from disk on every call."
        try:
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Reinitialize if the file was deleted.
            print("Warning: JSON file was deleted; recreating")
            self._initialize_or_load_json_file()
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"""Failed to read JSON file: {str(e)}""")
            raise

    def _save_json_data(self, data: Dict) -> None:
        "Save data to a JSON file."
        try:
            # Ensure the directory exists.
            os.makedirs(self.file_path, exist_ok=True)

            # Write atomically through a temporary file.
            temp_path = f"{self.full_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomically replace the destination.
            os.replace(temp_path, self.full_path)

        except Exception as e:
            # Clean up the temporary file.
            temp_path = f"{self.full_path}.tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"""Failed to save JSON file: {str(e)}""")
            raise


