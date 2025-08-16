import os
import sys
import cv2
import time
from pathlib import Path

from src.application.use_cases.frame_processing.async_frame_processor import AsyncFrameProcessor
from src.application.use_cases.video_processing.process_input_video import ProcessInputVideoUseCase
from src.application.use_cases.frame_processing.parallel_frame_processor import ParallelFrameProcessor
from src.application.use_cases.image_processing.find_matching_book_movie import FindMatchingBookMovieUseCase
from src.infrastructure.feature_extractors.sift_extractor import SIFTExtractor
from src.infrastructure.matchers.flann_matcher import FLANNMatcher
from src.infrastructure.repositories.file_image_repository import FileImageRepository
from src.infrastructure.repositories.file_video_repository import FileVideoRepository


def setup_test_environment():
    """Find project root and configure paths"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "main.py").exists() and (parent / "src").exists():
            sys.path.insert(0, str(parent))
            os.chdir(parent)
            return parent
    raise FileNotFoundError("Project root not found")


def parallel_processing():
    """Test parallel video processing"""
    print("🔄 Testing Parallel Video Processing...")

    setup_test_environment()

    # Verify directories
    for d in ["data/input_images", "data/book_movie_images", "data/trailers", "data/input_videos"]:
        assert os.path.isdir(d), f"Required directory missing: {d}"
        print(f"✅ Found directory: {d}")

    # Setup components
    feat_ext = SIFTExtractor()
    matcher = FLANNMatcher()
    img_repo = FileImageRepository()
    vid_repo = FileVideoRepository("data/trailers")

    # Create book matcher
    book_matcher = FindMatchingBookMovieUseCase(
        feature_extractor=feat_ext,
        matcher=matcher,
        image_repository=img_repo
    )

    # Create parallel frame processor
    parallel_processor = ParallelFrameProcessor(book_matcher, max_workers=6)

    # Create use case with parallel processor
    use_case = ProcessInputVideoUseCase(
        feature_extractor=feat_ext,
        matcher=matcher,
        image_repository=img_repo,
        video_repository=vid_repo,
        frame_processor=parallel_processor,
        min_conf=5.0
    )

    # Select test video
    videos = sorted(Path("data/input_videos").glob("*.mp4"))
    assert videos, "No input videos found in data/input_videos"
    test_vid = videos[3].name
    print(f"📹 Using test video: {test_vid}")

    # Progress callback
    def progress(msg, pct):
        print(f"  {pct:.1f}%: {msg}")

    # Execute parallel processing
    print("🚀 Starting parallel processing...")
    start_time = time.time()

    result = use_case.execute(
        input_video_name=test_vid,
        progress_callback=progress
    )

    parallel_time = time.time() - start_time

    # Assertions
    assert result.success, f"Parallel processing failed: {result.error_message}"
    assert result.total_frames_processed > 0, "No frames were processed"
    assert result.replaced_frames_count > 0, "No frames were replaced"
    assert os.path.exists(result.output_video_path), "Output video not created"

    # Verify output video properties
    cap_out = cv2.VideoCapture(result.output_video_path)
    out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cap_out.get(cv2.CAP_PROP_FPS)
    cap_out.release()

    assert out_frames == result.total_frames_processed, "Frame count mismatch in output"

    print(f"\n✅ Parallel Processing Test Passed!")
    print(f"  📖 Book detected: {result.target_book_name}")
    print(f"  🎬 Frames replaced: {result.replaced_frames_count}/{result.total_frames_processed}")
    print(f"  ⏱️ Processing time: {parallel_time:.1f} seconds")
    print(f"  🎥 Output FPS: {out_fps:.1f}")
    print(f"  💾 Output file: {result.output_video_path}")

    return result, parallel_time


def async_processing():
    """Test async video processing"""
    print("\n🔄 Testing Async Video Processing...")

    setup_test_environment()

    # Setup components
    feat_ext = SIFTExtractor()
    matcher = FLANNMatcher()
    img_repo = FileImageRepository()
    vid_repo = FileVideoRepository("data/trailers")

    # Create book matcher
    book_matcher = FindMatchingBookMovieUseCase(
        feature_extractor=feat_ext,
        matcher=matcher,
        image_repository=img_repo
    )

    # Create async frame processor
    async_processor = AsyncFrameProcessor(book_matcher, max_workers=6)

    # Create use case with async processor
    use_case = ProcessInputVideoUseCase(
        feature_extractor=feat_ext,
        matcher=matcher,
        image_repository=img_repo,
        video_repository=vid_repo,
        frame_processor=async_processor,
        min_conf=5.0
    )

    # Select test video
    videos = sorted(Path("data/input_videos").glob("*.mp4"))
    test_vid = videos[0].name
    print(f"📹 Using test video: {test_vid}")

    # Progress callback
    def progress(msg, pct):
        print(f"  {pct:.1f}%: {msg}")

    # Execute async processing
    print("🚀 Starting async processing...")
    start_time = time.time()

    result = use_case.execute(
        input_video_name=test_vid,
        progress_callback=progress
    )

    async_time = time.time() - start_time

    # Assertions
    assert result.success, f"Async processing failed: {result.error_message}"
    assert result.total_frames_processed > 0, "No frames were processed"
    assert result.replaced_frames_count > 0, "No frames were replaced"
    assert os.path.exists(result.output_video_path), "Output video not created"

    # Verify output video properties
    cap_out = cv2.VideoCapture(result.output_video_path)
    out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cap_out.get(cv2.CAP_PROP_FPS)
    cap_out.release()

    assert out_frames == result.total_frames_processed, "Frame count mismatch in output"

    print(f"\n✅ Async Processing Test Passed!")
    print(f"  📖 Book detected: {result.target_book_name}")
    print(f"  🎬 Frames replaced: {result.replaced_frames_count}/{result.total_frames_processed}")
    print(f"  ⏱️ Processing time: {async_time:.1f} seconds")
    print(f"  🎥 Output FPS: {out_fps:.1f}")
    print(f"  💾 Output file: {result.output_video_path}")

    return result, async_time


def test_both_processors():
    """Test both processors and compare performance"""
    print("🏁 Running Comprehensive Video Processing Test")
    print("=" * 60)

    try:
        # Test parallel processing
        parallel_result, parallel_time = parallel_processing()

        # Test async processing
        async_result, async_time = async_processing()

        # Compare results
        print(f"\n📊 Performance Comparison:")
        print(f"{'Method':<15} {'Time (s)':<10} {'Frames/s':<12} {'Book Detected':<15}")
        print("-" * 60)

        parallel_fps = parallel_result.total_frames_processed / parallel_time
        async_fps = async_result.total_frames_processed / async_time

        print(f"{'Parallel':<15} {parallel_time:<10.1f} {parallel_fps:<12.1f} {parallel_result.target_book_name:<15}")
        print(f"{'Async':<15} {async_time:<10.1f} {async_fps:<12.1f} {async_result.target_book_name:<15}")

        # Determine winner
        if parallel_time < async_time:
            winner = "Parallel"
            improvement = ((async_time - parallel_time) / async_time) * 100
        else:
            winner = "Async"
            improvement = ((parallel_time - async_time) / parallel_time) * 100

        print(f"\n🏆 Winner: {winner} processing ({improvement:.1f}% faster)")

        # Verify both produced valid outputs
        assert parallel_result.success and async_result.success, "Both methods should succeed"

        print(f"\n✅ All tests passed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_parallel():
    """Test only parallel processing (for quick testing)"""
    print("🔄 Quick Parallel Processing Test")
    print("=" * 40)

    try:
        result, processing_time = parallel_processing()

        print(f"\n📋 Summary:")
        print(f"  Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"  Processing time: {processing_time:.1f} seconds")
        print(f"  Frames per second: {result.total_frames_processed / processing_time:.1f}")

        return result.success

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def test_single_async():
    """Test only async processing (for quick testing)"""
    print("🔄 Quick Async Processing Test")
    print("=" * 40)

    try:
        result, processing_time = async_processing()

        print(f"\n📋 Summary:")
        print(f"  Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"  Processing time: {processing_time:.1f} seconds")
        print(f"  Frames per second: {result.total_frames_processed / processing_time:.1f}")

        return result.success

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test video replacement processing')
    parser.add_argument('--mode', choices=['parallel', 'async', 'both'],
                        default='both', help='Processing mode to test')

    args = parser.parse_args()

    if args.mode == 'parallel':
        success = test_single_parallel()
    elif args.mode == 'async':
        success = test_single_async()
    else:
        success = test_both_processors()

    sys.exit(0 if success else 1)
