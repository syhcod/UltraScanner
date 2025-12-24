#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h> // fft 라이브러리
#include "cam.h" // get_image() 포함. 

#define M_PI 3.14159265358979323846
#define GRID_SIZE 32
// 원본 이미지 크기
#define RAW_W IMAGE_WIDTH
#define RAW_H IMAGE_HEIGHT
// Green 추출 후 크기
#define PROC_W (IMAGE_WIDTH / 2)
#define PROC_H (IMAGE_HEIGHT / 2)
// 캔버스 크기 정의 (겹침 고려하여 넉넉하게 설정)
#define CANVAS_W (RAW_W * 10) 
#define CANVAS_H (RAW_H * 10)



typedef struct {
    unsigned char* data;
    int x_offset; // 전체 캔버스에서의 좌상단 x 좌표
    int y_offset; // 전체 캔버스에서의 좌상단 y 좌표
} ImageTile;

// 전역 캔버스 메모리
float* canvas_accum;
float* canvas_weight;

// 캔버스 메모리 초기화
void init_canvas() {
    canvas_accum = (float*)calloc(CANVAS_W * CANVAS_H, sizeof(float));
    canvas_weight = (float*)calloc(CANVAS_W * CANVAS_H, sizeof(float));
}

/**
 * 이미지를 캔버스에 블렌딩하여 추가
 * 선형 페더링(Linear Feathering) 기법 사용
 */
void blend_to_canvas(ImageTile* tile) {
    for (int y = 0; y < RAW_H; y++) {
        for (int x = 0; x < RAW_W; x++) {
            // 전체 캔버스 좌표 계산
            int cur_x = tile->x_offset + x;
            int cur_y = tile->y_offset + y;

            if (cur_x < 0 || cur_x >= CANVAS_W || cur_y < 0 || cur_y >= CANVAS_H) continue;

            // 가장자리 가중치 계산 (경계에서 0, 중심에서 1)
            // margin: 가중치가 줄어들기 시작하는 범위 (예: 이미지의 10%)
            float margin = RAW_W * 0.1f;
            float wx = 1.0f, wy = 1.0f;
            
            if (x < margin) wx = x / margin;
            else if (x > RAW_W - margin) wx = (RAW_W - x) / margin;
            
            if (y < margin) wy = y / margin;
            else if (y > RAW_H - margin) wy = (RAW_H - y) / margin;

            float weight = wx * wy; // 최종 2D 가중치
            if (weight < 0.01f) weight = 0.01f; // 0으로 나누기 방지

            int canvas_idx = cur_y * CANVAS_W + cur_x;
            canvas_accum[canvas_idx] += (float)tile->data[y * RAW_W + x] * weight;
            canvas_weight[canvas_idx] += weight;
        }
    }
}

/**
 * 최종 결과물 추출 (float 누적 데이터를 unsigned char로 변환)
 */
unsigned char* finalize_canvas() {
    unsigned char* result = (unsigned char*)malloc(CANVAS_W * CANVAS_H);
    for (int i = 0; i < CANVAS_W * CANVAS_H; i++) {
        if (canvas_weight[i] > 0) {
            float val = canvas_accum[i] / canvas_weight[i];
            result[i] = (unsigned char)(val > 255.0f ? 255 : val);
        } else {
            result[i] = 0;
        }
    }
    return result;
}

/**
 * Bayer 패턴(RGGB)에서 Green 채널 하나만 추출하고 
 * FFT 분석 전 경계 노이즈 제거를 위해 Hanning Window를 적용합니다.
 */
void preprocess_image(unsigned char* raw, double* output) {
    for (int y = 0; y < PROC_H; y++) {
        for (int x = 0; x < PROC_W; x++) {
            // RGGB 패턴 중 (0,1) 위치의 Green을 추출한다고 가정
            // raw 데이터 상의 인덱스 계산
            int raw_idx = (y * 2) * RAW_W + (x * 2 + 1);
            double pixel_val = (double)raw[raw_idx];

            // 2D Hanning Window 계산
            double wx = 0.5 * (1.0 - cos(2.0 * M_PI * x / (PROC_W - 1)));
            double wy = 0.5 * (1.0 - cos(2.0 * M_PI * y / (PROC_H - 1)));
            
            output[y * PROC_W + x] = pixel_val * wx * wy;
        }
    }
}

/**
 * 두 이미지의 겹치는 영역에서 위상 상관을 통해 변위를 계산
 * FFT(Img1) * conj(FFT(Img2)) / |...| -> IFFT -> Peak 위치 탐색
 */
void calculate_translation(unsigned char* img1, unsigned char* img2, int* dx, int* dy) {
    int N = PROC_W * PROC_H;
    double *proc1 = (double*)malloc(sizeof(double) * N);
    double *proc2 = (double*)malloc(sizeof(double) * N);

    // 1. 전처리 (Green 추출 + Windowing)
    preprocess_image(img1, proc1);
    preprocess_image(img2, proc2);

    fftw_complex *in1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *in2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *res = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *inv_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    // 복소수 배열로 복사
    for(int i=0; i<N; i++) {
        in1[i] = proc1[i] + 0 * I;
        in2[i] = proc2[i] + 0 * I;
    }

    // 2. FFT
    fftw_plan p1 = fftw_plan_dft_2d(PROC_H, PROC_W, in1, out1, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_2d(PROC_H, PROC_W, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_execute(p2);

    // 3. Normalized Cross-Power Spectrum
    for (int i = 0; i < N; i++) {
        fftw_complex temp = out1[i] * conj(out2[i]);
        double norm = cabs(temp);
        res[i] = (norm > 1e-9) ? (temp / norm) : 0;
    }

    // 4. Inverse FFT
    fftw_plan pi = fftw_plan_dft_2d(PROC_H, PROC_W, res, inv_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(pi);

    // 5. Peak Search
    double max_mag = -1;
    int peak_x = 0, peak_y = 0;
    for (int y = 0; y < PROC_H; y++) {
        for (int x = 0; x < PROC_W; x++) {
            double mag = cabs(inv_out[y * PROC_W + x]);
            if (mag > max_mag) {
                max_mag = mag;
                peak_x = x;
                peak_y = y;
            }
        }
    }

    // 주기성 처리 (Half-shift)
    if (peak_x > PROC_W / 2) peak_x -= PROC_W;
    if (peak_y > PROC_H / 2) peak_y -= PROC_H;

    // Green 추출 시 해상도가 절반이었으므로, 원래 이미지의 변위로 환산하려면 2를 곱함
    *dx = peak_x * 2;
    *dy = peak_y * 2;

    // 자원 해제
    free(proc1); free(proc2);
    fftw_destroy_plan(p1); fftw_destroy_plan(p2); fftw_destroy_plan(pi);
    fftw_free(in1); fftw_free(in2); fftw_free(out1); fftw_free(out2); fftw_free(res); fftw_free(inv_out);
}

int main() {
    init_canvas();
    ImageTile grid[GRID_SIZE][GRID_SIZE];

    // 시작점 설정 (캔버스의 중앙 부근)
    int start_x = CANVAS_W / 4;
    int start_y = CANVAS_H / 4;

    unsigned char **img = get_image(100);
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            grid[y][x].data = &(img[x][y]);

            if (x == 0 && y == 0) {
                grid[y][x].x_offset = start_x;
                grid[y][x].y_offset = start_y;
            } 
            else if (x > 0) {
                int dx, dy;
                calculate_translation(grid[y][x-1].data, grid[y][x].data, &dx, &dy);
                
                // [중요] Bayer 패턴 정렬을 위해 짝수로 맞춤
                dx = (dx / 2) * 2; 
                dy = (dy / 2) * 2;

                grid[y][x].x_offset = grid[y][x-1].x_offset + dx;
                grid[y][x].y_offset = grid[y][x-1].y_offset + dy;
            } 
            else if (y > 0) {
                // 줄이 바뀔 때 위쪽 타일과 비교
                int dx, dy;
                calculate_translation(grid[y-1][x].data, grid[y][x].data, &dx, &dy);
                
                dx = (dx / 2) * 2;
                dy = (dy / 2) * 2;

                grid[y][x].x_offset = grid[y-1][x].x_offset + dx;
                grid[y][x].y_offset = grid[y-1][x].y_offset + dy;
            }

            // 계산된 위치에 타일 합성
            blend_to_canvas(&grid[y][x]);
            
            // 메모리 절약을 위해 합성 후 데이터 해제 가능
            // free(grid[y][x].data); 
        }
    }

    unsigned char* final_img = finalize_canvas();
    // 이후 final_img 저장 로직...

    return 0;
}
