import cv2
import numpy as np
import json
from datetime import datetime


class ArucoMarkerDetector:
    """
    This class is used to detect aruco markers in the image and optimize detection parameters.
    """

    def __init__(self,
                 dictionary_id: int = cv2.aruco.DICT_ARUCO_ORIGINAL,
                 parameters: cv2.aruco.DetectorParameters = None):

        # Setup the detector - Updated for newer OpenCV versions
        try:
            # Try newer OpenCV API first (4.7+)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.parameters = parameters or cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        except AttributeError:
            # Fall back to older OpenCV API (4.6 and earlier)
            self.aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
            self.parameters = parameters or cv2.aruco.DetectorParameters_create()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.ground_plane_ids = {i for i in range(200, 204)}
        self.object_id = 204
        self.sensor_id = 205
        self.marker_ids = {*self.ground_plane_ids, self.object_id, self.sensor_id}

    def detect(self, image):
        """
        Detects aruco markers in an image

        Args:
            image (numpy.ndarray): The image to detect aruco markers in.

        Returns:
            tuple: (corners, ids, rejected) from ArUco detection
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected

    def vary_parameters(self, data):
        """Used to vary the parameters of the aruco detector to optimize detection"""

        # Import here to avoid circular imports
        from processing_capture import Processor

        # Get a sample capture to test with
        if not data.data_entries:
            print("No capture data available for testing")
            return

        # Use the first capture for testing
        test_capture = data.data_entries[0]
        processor = Processor(test_capture)
        test_image = processor.get_grayscale()

        if test_image is None:
            print("Could not get test image")
            return

        print(f"\n{'='*80}")
        print("TESTING ARUCO DETECTOR PARAMETERS")
        print(f"{'='*80}")
        print(f"Testing with capture 0, image shape: {test_image.shape}")

        # Default parameters as baseline
        default_params = cv2.aruco.DetectorParameters()

        # Test different parameter combinations
        parameter_tests = [
            {
                'name': 'Default Parameters',
                'params': default_params
            },
            {
                'name': 'Adaptive Threshold - Small Window',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'adaptiveThreshWinSizeMin': 3,
                    'adaptiveThreshWinSizeMax': 15,
                    'adaptiveThreshWinSizeStep': 2
                }
            },
            {
                'name': 'Adaptive Threshold - Large Window',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'adaptiveThreshWinSizeMin': 15,
                    'adaptiveThreshWinSizeMax': 45,
                    'adaptiveThreshWinSizeStep': 4
                }
            },
            {
                'name': 'High Threshold Constant',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'adaptiveThreshConstant': 10
                }
            },
            {
                'name': 'Low Threshold Constant',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'adaptiveThreshConstant': 3
                }
            },
            {
                'name': 'Relaxed Polygon Approximation',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'polygonalApproxAccuracyRate': 0.1
                }
            },
            {
                'name': 'Strict Polygon Approximation',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'polygonalApproxAccuracyRate': 0.01
                }
            },
            {
                'name': 'Small Marker Detection',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'minMarkerPerimeterRate': 0.01,
                    'maxMarkerPerimeterRate': 2.0
                }
            },
            {
                'name': 'Large Marker Detection',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'minMarkerPerimeterRate': 0.1,
                    'maxMarkerPerimeterRate': 4.0
                }
            },
            {
                'name': 'Corner Refinement - Subpixel',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
                    'cornerRefinementWinSize': 5,
                    'cornerRefinementMaxIterations': 30
                }
            },
            {
                'name': 'Corner Refinement - Contour',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_CONTOUR
                }
            },
            {
                'name': 'High Error Correction',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'errorCorrectionRate': 0.9
                }
            },
            {
                'name': 'Low Error Correction',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'errorCorrectionRate': 0.1
                }
            },
            {
                'name': 'Optimized Combo 1',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'adaptiveThreshWinSizeMin': 3,
                    'adaptiveThreshWinSizeMax': 23,
                    'adaptiveThreshConstant': 7,
                    'minMarkerPerimeterRate': 0.03,
                    'maxMarkerPerimeterRate': 4.0,
                    'polygonalApproxAccuracyRate': 0.05,
                    'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX
                }
            },
            {
                'name': 'Optimized Combo 2',
                'params': cv2.aruco.DetectorParameters(),
                'modifications': {
                    'adaptiveThreshWinSizeMin': 5,
                    'adaptiveThreshWinSizeMax': 21,
                    'adaptiveThreshConstant': 8,
                    'minMarkerPerimeterRate': 0.02,
                    'maxMarkerPerimeterRate': 3.0,
                    'polygonalApproxAccuracyRate': 0.03,
                    'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_CONTOUR,
                    'errorCorrectionRate': 0.6
                }
            }
        ]

        results = []
        best_results = []

        for test_config in parameter_tests:
            try:
                # Create parameters and apply modifications
                params = test_config['params']

                if 'modifications' in test_config:
                    for param_name, param_value in test_config['modifications'].items():
                        setattr(params, param_name, param_value)

                # Create detector with these parameters
                detector = ArucoMarkerDetector(cv2.aruco.DICT_ARUCO_ORIGINAL, params)
                corners, ids, rejected = detector.detect(test_image)

                num_markers = len(corners) if corners is not None else 0
                num_rejected = len(rejected) if rejected is not None else 0

                result = {
                    'name': test_config['name'],
                    'markers': num_markers,
                    'rejected': num_rejected,
                    'ids': ids.flatten().tolist() if ids is not None else [],
                    'params': params
                }
                results.append(result)

                # Print results
                status = "✅" if num_markers > 0 else "❌"
                print(f"{status} {test_config['name']:<35} | {num_markers:2d} markers | {num_rejected:2d} rejected | IDs: {result['ids']}")

                # Track best results
                if num_markers > 0:
                    best_results.append(result)

            except Exception as e:
                print(f"❌ {test_config['name']:<35} | ERROR: {e}")

        print(f"\n{'='*80}")
        print("PARAMETER OPTIMIZATION RESULTS")
        print(f"{'='*80}")

        if best_results:
            # Sort by number of markers detected (descending)
            best_results.sort(key=lambda x: x['markers'], reverse=True)

            print(f"Found {len(best_results)} parameter sets that detected markers:")
            for i, result in enumerate(best_results, 1):
                print(f"{i}. {result['name']}: {result['markers']} markers (IDs: {result['ids']})")

            # Show the best parameter set details
            best_result = best_results[0]
            print(f"\n{'='*80}")
            print(f"BEST PARAMETER SET: {best_result['name']}")
            print(f"{'='*80}")
            print(f"Markers detected: {best_result['markers']}")
            print(f"Marker IDs: {best_result['ids']}")
            print(f"Rejected candidates: {best_result['rejected']}")

            # Print the actual parameter values for the best result
            best_params = best_result['params']
            print(f"\nBest parameters:")
            print(f"  adaptiveThreshWinSizeMin: {best_params.adaptiveThreshWinSizeMin}")
            print(f"  adaptiveThreshWinSizeMax: {best_params.adaptiveThreshWinSizeMax}")
            print(f"  adaptiveThreshWinSizeStep: {best_params.adaptiveThreshWinSizeStep}")
            print(f"  adaptiveThreshConstant: {best_params.adaptiveThreshConstant}")
            print(f"  minMarkerPerimeterRate: {best_params.minMarkerPerimeterRate}")
            print(f"  maxMarkerPerimeterRate: {best_params.maxMarkerPerimeterRate}")
            print(f"  polygonalApproxAccuracyRate: {best_params.polygonalApproxAccuracyRate}")
            print(f"  cornerRefinementMethod: {best_params.cornerRefinementMethod}")
            print(f"  errorCorrectionRate: {best_params.errorCorrectionRate}")

            # Test best parameters on all captures
            print(f"\n{'='*80}")
            print(f"TESTING BEST PARAMETERS ON ALL CAPTURES")
            print(f"{'='*80}")

            detector = ArucoMarkerDetector(cv2.aruco.DICT_ARUCO_ORIGINAL, best_params)
            total_markers = 0

            for i, capture in enumerate(data.data_entries):
                processor = Processor(capture)
                image = processor.get_grayscale()
                if image is not None:
                    corners, ids, rejected = detector.detect(image)
                    num_markers = len(corners) if corners is not None else 0
                    marker_ids = ids.flatten().tolist() if ids is not None else []
                    total_markers += num_markers
                    print(f"Capture {i}: {num_markers} markers (IDs: {marker_ids})")

            print(f"\nTotal markers detected across all captures: {total_markers}")
            print(f"Average markers per capture: {total_markers / len(data.data_entries):.1f}")

            # Save the best parameters to a file
            config_filename = f"best_aruco_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Prepare the configuration data
            config_data = {
                "timestamp": datetime.now().isoformat(),
                "best_config_name": best_result['name'],
                "performance": {
                    "markers_detected": best_result['markers'],
                    "marker_ids": best_result['ids'],
                    "rejected_candidates": best_result['rejected'],
                    "total_markers_all_captures": total_markers,
                    "average_markers_per_capture": round(total_markers / len(data.data_entries), 1),
                    "num_captures_tested": len(data.data_entries)
                },
                "parameters": {
                    "dictionary": "cv2.aruco.DICT_ARUCO_ORIGINAL",
                    "adaptiveThreshWinSizeMin": int(best_params.adaptiveThreshWinSizeMin),
                    "adaptiveThreshWinSizeMax": int(best_params.adaptiveThreshWinSizeMax),
                    "adaptiveThreshWinSizeStep": int(best_params.adaptiveThreshWinSizeStep),
                    "adaptiveThreshConstant": float(best_params.adaptiveThreshConstant),
                    "minMarkerPerimeterRate": float(best_params.minMarkerPerimeterRate),
                    "maxMarkerPerimeterRate": float(best_params.maxMarkerPerimeterRate),
                    "polygonalApproxAccuracyRate": float(best_params.polygonalApproxAccuracyRate),
                    "cornerRefinementMethod": int(best_params.cornerRefinementMethod),
                    "cornerRefinementWinSize": int(best_params.cornerRefinementWinSize),
                    "cornerRefinementMaxIterations": int(best_params.cornerRefinementMaxIterations),
                    "errorCorrectionRate": float(best_params.errorCorrectionRate)
                },
                "usage_example": {
                    "python_code": f"""
# Load these parameters in your code:
import cv2
import json

# Load the saved parameters
with open('{config_filename}', 'r') as f:
    config = json.load(f)

# Create detector parameters
params = cv2.aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin = config['parameters']['adaptiveThreshWinSizeMin']
params.adaptiveThreshWinSizeMax = config['parameters']['adaptiveThreshWinSizeMax']
params.adaptiveThreshWinSizeStep = config['parameters']['adaptiveThreshWinSizeStep']
params.adaptiveThreshConstant = config['parameters']['adaptiveThreshConstant']
params.minMarkerPerimeterRate = config['parameters']['minMarkerPerimeterRate']
params.maxMarkerPerimeterRate = config['parameters']['maxMarkerPerimeterRate']
params.polygonalApproxAccuracyRate = config['parameters']['polygonalApproxAccuracyRate']
params.cornerRefinementMethod = config['parameters']['cornerRefinementMethod']
params.cornerRefinementWinSize = config['parameters']['cornerRefinementWinSize']
params.cornerRefinementMaxIterations = config['parameters']['cornerRefinementMaxIterations']
params.errorCorrectionRate = config['parameters']['errorCorrectionRate']

# Create detector
from ArucoMarkers import ArucoMarkerDetector
detector = ArucoMarkerDetector(cv2.aruco.DICT_ARUCO_ORIGINAL, params)
"""
                }
            }

            # Save to file
            try:
                with open(config_filename, 'w') as f:
                    json.dump(config_data, f, indent=2)
                print(f"\n✅ Best parameters saved to: {config_filename}")
                print(f"📋 You can load these parameters in future runs for optimal detection!")
            except Exception as e:
                print(f"❌ Error saving parameters to file: {e}")

        else:
            print("❌ No parameter sets detected any markers!")
            print("This could indicate:")
            print("1. No ArUco markers are present in the image")
            print("2. The markers are severely distorted or corrupted")
            print("3. The image quality is too poor for detection")
            print("4. The markers are from a different dictionary")

        print(f"{'='*80}")
        return best_results

    def test_dictionaries(self, data):
        """Used to determine which dictionaries the Aruco Markers are from"""

        # Import here to avoid circular imports
        from processing_capture import Processor

        # Get a sample capture to test with
        if not data.data_entries:
            print("No capture data available for testing")
            return

        # Use the first capture for testing
        test_capture = data.data_entries[0]
        processor = Processor(test_capture)
        test_image = processor.get_grayscale()

        if test_image is None:
            print("Could not get test image")
            return

        # List of all available ArUco dictionaries
        dictionaries = [
            ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
            ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
            ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
            ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
            ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
            ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
            ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
            ("DICT_5X5_1000", cv2.aruco.DICT_5X5_1000),
            ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
            ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
            ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
            ("DICT_6X6_1000", cv2.aruco.DICT_6X6_1000),
            ("DICT_7X7_50", cv2.aruco.DICT_7X7_50),
            ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
            ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
            ("DICT_7X7_1000", cv2.aruco.DICT_7X7_1000),
            ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
            ("DICT_APRILTAG_16h5", cv2.aruco.DICT_APRILTAG_16h5),
            ("DICT_APRILTAG_25h9", cv2.aruco.DICT_APRILTAG_25h9),
            ("DICT_APRILTAG_36h10", cv2.aruco.DICT_APRILTAG_36h10),
            ("DICT_APRILTAG_36h11", cv2.aruco.DICT_APRILTAG_36h11),
        ]

        print(f"\n{'='*80}")
        print("TESTING ALL ARUCO DICTIONARIES")
        print(f"{'='*80}")
        print(f"Testing with capture 0, image shape: {test_image.shape}")

        results = []
        best_results = []

        for dict_name, dict_id in dictionaries:
            try:
                # Create detector with this dictionary
                detector = ArucoMarkerDetector(dict_id)
                corners, ids, rejected = detector.detect(test_image)

                num_markers = len(corners) if corners is not None else 0
                num_rejected = len(rejected) if rejected is not None else 0

                result = {
                    'name': dict_name,
                    'markers': num_markers,
                    'rejected': num_rejected,
                    'ids': ids.flatten().tolist() if ids is not None else [],
                    'dict_id': dict_id
                }
                results.append(result)

                # Print results
                status = "✅" if num_markers > 0 else "❌"
                print(f"{status} {dict_name:<25} | {num_markers:2d} markers | {num_rejected:2d} rejected | IDs: {result['ids']}")

                # Track best results
                if num_markers > 0:
                    best_results.append(result)

            except Exception as e:
                print(f"❌ {dict_name:<25} | ERROR: {e}")

        print(f"\n{'='*80}")
        print("SUMMARY OF BEST RESULTS")
        print(f"{'='*80}")

        if best_results:
            # Sort by number of markers detected (descending)
            best_results.sort(key=lambda x: x['markers'], reverse=True)

            print(f"Found {len(best_results)} dictionaries that detected markers:")
            for i, result in enumerate(best_results, 1):
                print(f"{i}. {result['name']}: {result['markers']} markers (IDs: {result['ids']})")

            # Test the best dictionary on all captures
            best_dict = best_results[0]
            print(f"\n{'='*80}")
            print(f"TESTING BEST DICTIONARY ({best_dict['name']}) ON ALL CAPTURES")
            print(f"{'='*80}")

            detector = ArucoMarkerDetector(best_dict['dict_id'])

            for i, capture in enumerate(data.data_entries):
                processor = Processor(capture)
                image = processor.get_grayscale()
                if image is not None:
                    corners, ids, rejected = detector.detect(image)
                    num_markers = len(corners) if corners is not None else 0
                    marker_ids = ids.flatten().tolist() if ids is not None else []
                    print(f"Capture {i}: {num_markers} markers (IDs: {marker_ids})")

            # Show combined results
            print(f"\n{'='*80}")
            print("RECOMMENDATION")
            print(f"{'='*80}")
            print(f"Best dictionary: {best_dict['name']}")
            print(f"To use this dictionary, change the ArucoMarkerDetector initialization to:")
            print(f"from ArucoMarkers import ArucoMarkerDetector")
            print(f"detector = ArucoMarkerDetector(cv2.aruco.{best_dict['name']})")

        else:
            print("❌ No dictionaries detected any markers!")
            print("This could mean:")
            print("1. No ArUco markers are present in the image")
            print("2. The markers are too small, blurry, or distorted")
            print("3. The markers are from a custom dictionary not tested")

        print(f"{'='*80}")
        return best_results
