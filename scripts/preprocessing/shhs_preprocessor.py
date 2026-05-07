"""
SHHS (Sleep Heart Health Study) Preprocessing Pipeline
Efficient, disk-space-conscious preprocessing for sleep apnea detection

This pipeline:
1. Downloads SHHS data using NSRR token
2. Extracts only needed signals (Airflow + SpO2)
3. Processes in streaming mode (minimal disk usage)
4. Outputs data compatible with train_with_manual_data.py

Usage:
    python shhs_preprocessor.py --token YOUR_NSRR_TOKEN --max-subjects 10

    token: 30222-jzBS8hQqhdxaazCzHstq
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
import requests
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import pyedflib as edf


class SHHSPreprocessor:
    """
    SHHS Dataset Preprocessor
    Downloads and processes SHHS data efficiently
    """
    
    def __init__(self, nsrr_token: str, cache_dir: str = "data/shhs_processed"):
        """
        Args:
            nsrr_token: Your NSRR API token from sleepdata.org
            cache_dir: Where to save processed data (only processed data, not raw)
        """
        self.token = nsrr_token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # SHHS NSRR URLs
        self.base_url = "https://sleepdata.org/datasets/shhs/files"
        self.api_headers = {"Accept": "application/json"}
        
        # Signal configuration
        self.target_signals = ['AIRFLOW', 'SAO2', 'H.R.']  # Airflow, SpO2, Heart Rate
        self.target_fs = 50  # Resample to 50 Hz
        self.window_size = 30  # 30-second windows
        self.window_step = 15  # 50% overlap
        
    def get_subject_list(self, max_subjects: Optional[int] = None) -> List[str]:
        """
        Get list of available SHHS subjects
        
        Args:
            max_subjects: Maximum number of subjects (None for all)
        
        Returns:
            List of subject IDs
        """
        # SHHS naming: shhs1-200001 to shhs1-202903 (Visit 1)
        # For efficiency, we'll use a subset
        all_subjects = []
        
        # SHHS1 subjects (first visit)
        for i in range(200001, 200401):  # First 100 subjects
            all_subjects.append(f"shhs2-{i}")
        
        if max_subjects:
            return all_subjects[:max_subjects]
        return all_subjects
    
    def download_file(self, subject_id: str, file_type: str) -> Optional[Path]:
        """
        Download a file from NSRR
        
        Args:
            subject_id: Subject ID (e.g., 'shhs1-200001')
            file_type: 'edf' or 'xml'
        
        Returns:
            Path to downloaded file (in temp directory)
        """
        # Construct URL
        if file_type == 'edf':
            url = f"{self.base_url}/polysomnography/edfs/shhs2/{subject_id}.edf.gz"
        elif file_type == 'xml':
            url = f"{self.base_url}/polysomnography/annotations-events-nsrr/shhs2/{subject_id}-nsrr.xml.gz"
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        try:
            # Download to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}')

            response = requests.get(
                url + f"?download=1",
                headers={
                    "Authorization": f"Token {self.token}",
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/octet-stream"
                },
                stream=True
            )
            print(f"  Status: {response.status_code}")
            print(f"  Content-Type: {response.headers.get('content-type')}")
            if "text/html" in response.headers.get("content-type", ""):
                print("  ✗ Received HTML instead of file (authentication failed or access not granted)")
                return None
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            print(f"  Download size: {total_size / (1024*1024):.2f} MB")

            # Stream download to save memory
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

            temp_file.close()

            # Check if file is gzipped and decompress if needed
            with open(temp_file.name, 'rb') as f:
                header = f.read(2)
                if header == b'\x1f\x8b':
                    # Gzipped file, decompress
                    import gzip
                    decompressed = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}')
                    with gzip.open(temp_file.name, 'rb') as gz:
                        decompressed.write(gz.read())
                    decompressed.close()
                    os.unlink(temp_file.name)
                    return Path(decompressed.name)
                else:
                    return Path(temp_file.name)

        except Exception as e:
            print(f"  Error downloading {file_type} for {subject_id}: {e}")
            return None
    
    def extract_signals(self, edf_path: Path) -> Dict[str, np.ndarray]:
        """
        Extract only needed signals from EDF file using pyedflib
        """
        try:
            with edf.EdfReader(str(edf_path)) as reader:
                labels = reader.getSignalLabels()
                print("  Available signals:", labels)
                n_signals = reader.signals_in_file
                fs = reader.getSampleFrequencies()

                # Find target signal indices
                signal_indices = {}
                for i, label in enumerate(labels):
                    label_u = (label or '').upper()
                    if 'AIRFLOW' in label_u or 'NEW AIR' in label_u or 'THOR RES' in label_u or 'ABDO RES' in label_u:
                        signal_indices['AIRFLOW'] = i
                    elif 'SAO2' in label_u or 'SPO2' in label_u:
                        signal_indices['SAO2'] = i
                    elif 'H.R.' in label_u or 'HR' in label_u or 'PULSE' in label_u:
                        signal_indices['HR'] = i

                if not signal_indices:
                    return {}

                # Extract signals
                signals = {}
                for sig_name, sig_idx in signal_indices.items():
                    data = reader.readSignal(sig_idx)
                    signals[sig_name] = data
                    signals[f'{sig_name}_fs'] = fs[sig_idx]

                return signals
        except Exception as e:
            print(f"  Error reading EDF file: {e}")
            return {}
    
    def parse_annotations(self, xml_path: Path) -> List[Dict]:
        """Parse SHHS XML annotations"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            events = []
            
            for event in root.findall('.//ScoredEvent'):
                # SHHS NSRR XML uses <Name> element (not <EventType>)
                event_type_elem = event.find('Name') or event.find('EventType') or event.find('EventConcept')
                if event_type_elem is None or event_type_elem.text is None:
                    continue
                event_type = event_type_elem.text.strip()
                if not event_type:
                    continue

                # Only keep apnea/hypopnea events
                if not any(keyword in event_type.upper() for keyword in ['APNEA', 'HYPOPNEA']):
                    continue
                
                start_time = float(event.find('Start').text)
                duration = float(event.find('Duration').text)
                
                # Try to get SpO2 info
                spo2_nadir = None
                spo2_baseline = None
                
                spo2_nadir_elem = event.find('SpO2Nadir')
                if spo2_nadir_elem is not None and spo2_nadir_elem.text:
                    try:
                        spo2_nadir = float(spo2_nadir_elem.text)
                    except:
                        pass
                
                spo2_baseline_elem = event.find('SpO2Baseline')
                if spo2_baseline_elem is not None and spo2_baseline_elem.text:
                    try:
                        spo2_baseline = float(spo2_baseline_elem.text)
                    except:
                        pass
                
                events.append({
                    'type': event_type,
                    'start': start_time,
                    'duration': duration,
                    'end': start_time + duration,
                    'spo2_nadir': spo2_nadir,
                    'spo2_baseline': spo2_baseline
                })
            
            return events
            
        except Exception as e:
            print(f"  Error parsing annotations: {e}")
            return []
    
    def preprocess_signals(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess signals: filter, resample, extract features
        """
        processed = {}
        
        # Process Airflow → Respiration Rate
        if 'AIRFLOW' in signals:
            airflow = signals['AIRFLOW']
            fs_original = signals['AIRFLOW_fs']
            
            # Bandpass filter (0.1-3 Hz for respiratory frequency)
            nyquist = fs_original / 2
            low = 0.1 / nyquist
            high = 3.0 / nyquist
            b, a = butter(4, [low, high], btype='band')
            airflow_filtered = filtfilt(b, a, airflow)
            
            # Detect breathing peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(airflow_filtered, distance=int(2 * fs_original))
            
            if len(peaks) > 1:
                breath_intervals = np.diff(peaks) / fs_original
                resp_rate = 60.0 / breath_intervals
                resp_rate = np.clip(resp_rate, 5, 40)
                
                # Interpolate to 1 Hz
                time_peaks = peaks[1:] / fs_original
                time_uniform = np.arange(0, len(airflow) / fs_original, 1.0)
                resp_uniform = np.interp(time_uniform, time_peaks, resp_rate)
                
                processed['respiration'] = resp_uniform
            else:
                # Fallback: constant respiration
                processed['respiration'] = np.ones(int(len(airflow) / fs_original)) * 16.0
        
        # Process SpO2
        if 'SAO2' in signals:
            spo2 = signals['SAO2']
            fs_original = signals['SAO2_fs']
            
            # Clean invalid values
            spo2_clean = np.where(spo2 < 70, np.nan, spo2)
            spo2_clean = np.where(spo2_clean > 100, np.nan, spo2_clean)
            
            # Interpolate missing values
            mask = ~np.isnan(spo2_clean)
            if np.any(mask):
                indices = np.arange(len(spo2_clean))
                spo2_clean = np.interp(indices, indices[mask], spo2_clean[mask])
            
            # Resample to 1 Hz
            n_samples = int(len(spo2) / fs_original)
            spo2_resampled = resample(spo2_clean, n_samples)
            
            processed['spo2'] = spo2_resampled
        
        # Process Heart Rate
        if 'HR' in signals:
            hr = signals['HR']
            fs_original = signals['HR_fs']

            # Resample to 1 Hz
            n_samples = int(len(hr) / fs_original)
            hr_resampled = resample(hr, n_samples)
            hr_resampled = np.clip(hr_resampled, 30, 200)

            processed['heart_rate'] = hr_resampled
        
        return processed
    
    def create_labels(self, events: List[Dict], duration: float) -> np.ndarray:
        """Create binary labels at 1 Hz"""
        labels = np.zeros(int(duration), dtype=np.int32)
        
        for event in events:
            start = int(event['start'])
            end = int(event['end'])
            
            if start < len(labels):
                end = min(end, len(labels))
                labels[start:end] = 1
        
        return labels
    
    def process_subject(self, subject_id: str) -> Optional[Dict]:
        """
        Process a single subject
        Downloads EDF and XML, extracts signals, cleans up temp files
        
        Returns:
            Dictionary compatible with train_with_manual_data.py format
        """
        print(f"\nProcessing {subject_id}...")
        delete_after = False

        # Check for local files first (recommended to avoid NSRR auth issues)
        local_edf = Path(f"data/raw/{subject_id}.edf")
        local_xml = Path(f"data/raw/{subject_id}-nsrr.xml")

        if local_edf.exists() and local_xml.exists():
            print("  ✓ Using local files (bypassing download)")
            edf_path = local_edf
            xml_path = local_xml
            delete_after = False  # keep local files
        else:
            # Download files
            edf_path = self.download_file(subject_id, 'edf')
            xml_path = self.download_file(subject_id, 'xml')

        if not edf_path or not xml_path:
            return None
        
        try:
            # Extract signals (memory efficient)
            raw_signals = self.extract_signals(edf_path)
            print(f"  Raw signal keys: {list(raw_signals.keys())}")

            if not raw_signals:
                print(f"  ✗ No target signals found")
                return None

            print(f"  ✓ Extracted {len([k for k in raw_signals.keys() if '_fs' not in k])} signals")

            # Parse annotations
            events = self.parse_annotations(xml_path)
            print(f"  ✓ Found {len(events)} apnea events")

            # Preprocess signals
            processed_signals = self.preprocess_signals(raw_signals)
            print(f"  Processed signal keys: {list(processed_signals.keys())}")

            # Estimate duration
            first_signal = next(iter(processed_signals.values()))
            duration = len(first_signal)

            # Create labels
            labels = self.create_labels(events, duration)

            apnea_time = np.sum(labels)
            apnea_pct = (apnea_time / len(labels)) * 100

            print(f"  ✓ Processed signals to 1 Hz ({duration} seconds)")
            print(f"  ✓ Apnea: {apnea_time}s ({apnea_pct:.1f}%)")

            # Format output (compatible with your training code)
            result = {
                'subject_id': subject_id,
                'signals': processed_signals,
                'labels': labels,
                'events': events,
                'duration': float(duration)
            }

            return result

        finally:
            # Delete temp files OR local raw files if flagged
            if edf_path and edf_path.exists():
                if "tmp" in str(edf_path) or delete_after:
                    os.unlink(edf_path)
            if xml_path and xml_path.exists():
                if "tmp" in str(xml_path) or delete_after:
                    os.unlink(xml_path)
    
    def process_dataset(self, max_subjects: Optional[int] = None) -> List[Dict]:
        """
        Process multiple subjects from SHHS
        
        Args:
            max_subjects: Maximum number of subjects to process
        
        Returns:
            List of processed subject data
        """
        subjects = self.get_subject_list(max_subjects)
        
        print("\n" + "="*70)
        print(f"SHHS Dataset Preprocessing")
        print(f"Subjects to process: {len(subjects)}")
        print("="*70)
        
        all_data = []
        successful = 0
        
        for subject_id in tqdm(subjects, desc="Processing subjects"):
            data = self.process_subject(subject_id)
            
            if data:
                # Check if has required signals
                required_signals = ['heart_rate', 'respiration']
                if all(sig in data['signals'] for sig in required_signals):
                    all_data.append(data)
                    successful += 1
                else:
                    print(f"  ⚠ Missing required signals")
        
        print("\n" + "="*70)
        print(f"Successfully processed: {successful}/{len(subjects)} subjects")
        print("="*70)
        
        return all_data
    
    def save_processed_data(self, all_data: List[Dict], output_file: str = "shhs_processed.npz"):
        """
        Save processed data in compact format, appending to existing file if present
        
        Args:
            all_data: List of processed subjects
            output_file: Output filename
        """
        output_path = self.cache_dir / output_file
        
        # Prepare data for saving
        X_list = []
        y_list = []
        subject_ids = []
        
        for subject_data in all_data:
            signals = subject_data['signals']
            labels = subject_data['labels']
            
            # Stack signals
            min_length = min(len(signals[k]) for k in signals.keys())
            min_length = min(min_length, len(labels))
            
            hr = signals['heart_rate'][:min_length]
            resp = signals['respiration'][:min_length]
            spo2 = signals.get('spo2', np.random.normal(97, 1, min_length))[:min_length]
            
            X = np.stack([hr, resp, spo2], axis=0)
            
            X_list.append(X)
            y_list.append(labels[:min_length])
            subject_ids.append(subject_data['subject_id'])
        
        # Load existing data if file exists
        if output_path.exists():
            existing_data = np.load(output_path, allow_pickle=True)
            existing_X = [np.array(x) for x in existing_data['X']]
            existing_y = [np.array(y) for y in existing_data['y']]
            existing_ids = list(existing_data['subject_ids'])
            
            X_list = existing_X + X_list
            y_list = existing_y + y_list
            subject_ids = existing_ids + subject_ids
        
        # Pad arrays to same length
        if X_list:
            max_length = max(len(x[0]) for x in X_list)
            for i in range(len(X_list)):
                current_len = X_list[i].shape[1]
                if current_len < max_length:
                    pad_width = ((0, 0), (0, max_length - current_len))
                    X_list[i] = np.pad(X_list[i], pad_width, mode='constant', constant_values=0)
                    y_list[i] = np.pad(y_list[i], (0, max_length - current_len), mode='constant', constant_values=0)
        
        # Save
        np.savez_compressed(
            output_path,
            X=X_list,
            y=y_list,
            subject_ids=subject_ids
        )
        
        print(f"\n✓ Saved processed data to {output_path}")
        print(f"  Size on disk: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


# ==================== INTEGRATION WITH EXISTING TRAINING CODE ====================

class SHHSDatasetLoader:
    """
    Loader that mimics the interface of ManualUCDLoader
    Works with train_with_manual_data.py
    """
    
    def __init__(self, processed_file: str = "data/shhs_processed/shhs_processed.npz"):
        """
        Args:
            processed_file: Path to preprocessed SHHS data
        """
        self.processed_file = Path(processed_file)
        
        if not self.processed_file.exists():
            raise FileNotFoundError(
                f"Processed data not found: {processed_file}\n"
                f"Run preprocessing first:\n"
                f"  python shhs_preprocessor.py --token YOUR_TOKEN --max-subjects 10"
            )
        
        # Load data
        data = np.load(self.processed_file, allow_pickle=True)
        self.X_list = data['X']
        self.y_list = data['y']
        self.subject_ids = data['subject_ids']
        
        print(f"Loaded {len(self.X_list)} subjects from {processed_file}")
    
    def load_all_subjects(self, max_subjects: Optional[int] = None) -> List[Dict]:
        """
        Return data in format compatible with train_with_manual_data.py
        """
        subjects_to_use = min(max_subjects, len(self.X_list)) if max_subjects else len(self.X_list)
        
        all_data = []
        
        for i in range(subjects_to_use):
            X = self.X_list[i]
            y = self.y_list[i]
            
            # Convert to expected format
            data = {
                'subject_id': str(self.subject_ids[i]),
                'signals': {
                    'heart_rate': X[0],
                    'respiration': X[1],
                    'spo2': X[2]
                },
                'labels': y,
                'duration': float(len(y))
            }
            
            all_data.append(data)
        
        return all_data
    
    def prepare_for_training(self, all_data: List[Dict]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Prepare for training (same interface as ManualUCDLoader)"""
        X_list = []
        y_list = []
        
        for subject_data in all_data:
            signals = subject_data['signals']
            labels = subject_data['labels']
            
            # Stack signals
            X = np.stack([
                signals['heart_rate'],
                signals['respiration'],
                signals['spo2']
            ], axis=0)
            
            X_list.append(X)
            y_list.append(labels)
        
        return X_list, y_list


# ==================== COMMAND LINE INTERFACE ====================

def main():
    parser = argparse.ArgumentParser(description='SHHS Dataset Preprocessor')
    parser.add_argument('--token', required=True, help='NSRR API token from sleepdata.org')
    parser.add_argument('--max-subjects', type=int, default=10, 
                       help='Maximum number of subjects to process (default: 10)')
    parser.add_argument('--output-dir', default='data/shhs_processed',
                       help='Output directory for processed data')
    parser.add_argument('--output-file', default='auto',
                       help='Output filename (default auto: shhs_processed_{first}_{last}.npz)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = SHHSPreprocessor(
        nsrr_token=args.token,
        cache_dir=args.output_dir
    )

    # Determine output filename
    subjects = preprocessor.get_subject_list(args.max_subjects)
    if not subjects:
        print("No subjects found for max_subjects", args.max_subjects)
        return

    if args.output_file in (None, '', 'auto'):
        output_file = f"shhs_processed_{subjects[0]}_{subjects[-1]}.npz"
    else:
        output_file = args.output_file

    print(f"Output file set to: {output_file}")
    
    # Process data
    all_data = preprocessor.process_dataset(max_subjects=args.max_subjects)
    
    if all_data:
        # Save processed data
        preprocessor.save_processed_data(all_data, output_file)
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE!")
        print("="*70)
        print(f"\nTo use with your training code:")
        print(f"  1. Update train_with_manual_data.py to use SHHSDatasetLoader")
        print(f"  2. Or run: python train_with_shhs.py")
        print("\n" + "="*70)
    else:
        print("\n✗ No data was successfully processed")


if __name__ == "__main__":
    main()
