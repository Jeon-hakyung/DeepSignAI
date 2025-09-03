import numpy as np
import time
import cv2
import os

try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("🚀 GPU 테스트 모드")
except:
    GPU_AVAILABLE = False
    print("💻 CPU 전용 모드")


def cpu_calculation(data_arrays, iterations=1000):
    """CPU 벡터 연산"""
    start_time = time.time()

    results = []
    for _ in range(iterations):
        for arr in data_arrays:
            # 각도 계산 시뮬레이션
            v1 = arr[:3] - arr[3:6]
            v2 = arr[6:9] - arr[3:6]

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                results.append(np.degrees(angle))

    end_time = time.time()
    return end_time - start_time, len(results)


def gpu_calculation(data_arrays, iterations=1000):
    """GPU 벡터 연산"""
    if not GPU_AVAILABLE:
        return float('inf'), 0

    start_time = time.time()

    results = []
    for _ in range(iterations):
        for arr in data_arrays:
            try:
                # CPU → GPU
                arr_gpu = cp.asarray(arr)

                # GPU 연산
                v1 = arr_gpu[:3] - arr_gpu[3:6]
                v2 = arr_gpu[6:9] - arr_gpu[3:6]

                norm1 = cp.linalg.norm(v1)
                norm2 = cp.linalg.norm(v2)

                if norm1 > 0 and norm2 > 0:
                    cos_angle = cp.dot(v1, v2) / (norm1 * norm2)
                    angle = cp.arccos(cp.clip(cos_angle, -1, 1))
                    # GPU → CPU
                    result = float(cp.degrees(angle))
                    results.append(result)
            except:
                # GPU 오류 시 건너뛰기
                pass

    end_time = time.time()
    return end_time - start_time, len(results)


def test_video_processing_speed():
    """실제 비디오 처리 속도 테스트"""
    print("\n🎥 비디오 처리 속도 테스트")
    print("=" * 50)

    # 테스트용 더미 비디오 생성
    frames = []
    for i in range(100):  # 100 프레임
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)

    # CPU 처리 시간
    start_cpu = time.time()
    cpu_results = []
    for frame in frames:
        # OpenCV 색상 변환 (실제 처리)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 더미 특징 추출
        features = np.random.random(244).tolist()
        cpu_results.append(features)
    cpu_time = time.time() - start_cpu

    # GPU 처리 시간 (CuPy 포함)
    start_gpu = time.time()
    gpu_results = []
    for frame in frames:
        # OpenCV 색상 변환 (여전히 CPU)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # GPU 연산 시뮬레이션
        if GPU_AVAILABLE:
            try:
                dummy_data = np.random.random(100)
                gpu_data = cp.asarray(dummy_data)
                gpu_result = cp.sum(gpu_data)  # 간단한 GPU 연산
                features = [float(gpu_result)] * 244
            except:
                features = np.random.random(244).tolist()
        else:
            features = np.random.random(244).tolist()
        gpu_results.append(features)
    gpu_time = time.time() - start_gpu

    print(f"💻 CPU 처리: {cpu_time:.3f}초 ({100 / cpu_time:.1f} FPS)")
    print(f"🚀 GPU 처리: {gpu_time:.3f}초 ({100 / gpu_time:.1f} FPS)")

    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"✅ GPU가 {speedup:.1f}배 빠름!")
    else:
        slowdown = gpu_time / cpu_time
        print(f"❌ GPU가 {slowdown:.1f}배 느림")

    return cpu_time, gpu_time


def main():
    print("🎯 CPU vs GPU 성능 비교 테스트")
    print("=" * 50)

    # 테스트 데이터 생성 (손동작 시뮬레이션)
    data_size = 50  # 프레임 수
    data_arrays = []

    for _ in range(data_size):
        # 3D 포인트 9개 (각도 계산용)
        points = np.random.random(9).astype(np.float32)
        data_arrays.append(points)

    print(f"📊 테스트 데이터: {data_size}개 배열")
    print("🧮 연산: 벡터 각도 계산 (손동작 특징과 유사)")

    # CPU 테스트
    print("\n💻 CPU 테스트 중...")
    cpu_time, cpu_count = cpu_calculation(data_arrays, iterations=100)

    # GPU 테스트
    print("🚀 GPU 테스트 중...")
    gpu_time, gpu_count = gpu_calculation(data_arrays, iterations=100)

    # 결과 비교
    print("\n📊 벡터 연산 결과:")
    print(f"💻 CPU: {cpu_time:.3f}초 ({cpu_count}개 연산)")
    print(f"🚀 GPU: {gpu_time:.3f}초 ({gpu_count}개 연산)")

    if gpu_time < cpu_time and gpu_count > 0:
        speedup = cpu_time / gpu_time
        print(f"✅ GPU가 {speedup:.1f}배 빠름!")
    elif gpu_count == 0:
        print("❌ GPU 연산 실패 - CuPy 문제")
    else:
        slowdown = gpu_time / cpu_time
        print(f"❌ GPU가 {slowdown:.1f}배 느림 (오버헤드)")

    # 실제 비디오 처리 테스트
    test_video_processing_speed()

    # 결론
    print("\n🎯 결론:")
    if GPU_AVAILABLE:
        print("- GPU는 대용량 데이터에 적합")
        print("- 손동작 데이터는 크기가 작아서 CPU가 더 효율적일 수 있음")
        print("- MediaPipe/OpenCV가 CPU에서 실행되어 전체적으로는 CPU가 병목")
        print("- 실제 성능 향상을 원한다면 CPU 최적화 추천")
    else:
        print("- CuPy 설치 후 다시 테스트 필요")

    print("\n💡 권장사항:")
    print("- 현재 작업에는 CPU 버전이 더 실용적")
    print("- GPU는 딥러닝 모델 학습/추론에 사용하는 것이 효과적")


if __name__ == "__main__":
    main()