#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "cam.h" 

using namespace std;
using namespace cv;

#define GRID_SIZE 32
#define RAW_W IMAGE_WIDTH
#define RAW_H IMAGE_HEIGHT

// 캔버스 크기: 32x32 그리드, 여유 있게 설정
#define CANVAS_W (RAW_W * 35)
#define CANVAS_H (RAW_H * 35)

Mat get_opencv_image(int index) {
    unsigned char** raw_2d = get_image(100);
    if (!raw_2d) return Mat();
    Mat img(RAW_H, RAW_W, CV_8UC1);
    for (int y = 0; y < RAW_H; y++) {
        memcpy(img.ptr<uchar>(y), raw_2d[y], RAW_W);
        free(raw_2d[y]);
    }
    free(raw_2d);
    return img;
}

int main() {
    Mat canvas_accum = Mat::zeros(CANVAS_H, CANVAS_W, CV_32FC1);
    Mat canvas_weight = Mat::zeros(CANVAS_H, CANVAS_W, CV_32FC1);

    Point2d pos_grid[GRID_SIZE][GRID_SIZE];
    Mat row_starter_images[GRID_SIZE]; // 각 행의 첫 번째 이미지를 저장 (수직 정합용)

    // 1. 초기 시작점 (캔버스 중앙 부근)
    Point2d current_anchor(CANVAS_W / 2.0, CANVAS_H / 2.0);

    Mat hann;
    createHanningWindow(hann, Size(RAW_W, RAW_H), CV_64F);

    // --- 위에서 아래로 (y) ---
    for (int y = 0; y < GRID_SIZE; y++) {
        Mat left_tile_img; // 왼쪽 타일 저장용 (수평 정합용)

        // --- 좌에서 우로 (x) ---
        for (int x = 0; x < GRID_SIZE; x++) {
            Mat curr_img = get_opencv_image(y * GRID_SIZE + x);
            if (curr_img.empty()) continue;

            Mat curr_float;
            curr_img.convertTo(curr_float, CV_64F);

            if (x == 0 && y == 0) {
                // [Case 1] 전체 그리드의 시작점 (0,0)
                pos_grid[y][x] = current_anchor;
                row_starter_images[y] = curr_float;
            } 
            else if (x == 0) {
                // [Case 2] 새로운 행의 시작: 위쪽 행의 첫 번째 타일(y-1, 0)과 비교
                double response;
                Point2d shift = phaseCorrelate(row_starter_images[y-1], curr_float, hann, &response);
                
                // 위쪽 타일 좌표에 상대적 이동량(shift) 더하기
                pos_grid[y][x] = pos_grid[y-1][x] + shift;
                row_starter_images[y] = curr_float; // 다음 행을 위해 저장
            } 
            else {
                // [Case 3] 행 내부 이동: 바로 왼쪽 타일(y, x-1)과 비교
                double response;
                Point2d shift = phaseCorrelate(left_tile_img, curr_float, hann, &response);
                
                // 왼쪽 타일 좌표에 상대적 이동량(shift) 더하기
                pos_grid[y][x] = pos_grid[y][x-1] + shift;
            }

            // 2. 캔버스에 블렌딩
            int ox = cvRound(pos_grid[y][x].x);
            int oy = cvRound(pos_grid[y][x].y);

            float margin = RAW_W * 0.15f;
            for (int i = 0; i < RAW_H; i++) {
                int cy = oy + i;
                if (cy < 0 || cy >= CANVAS_H) continue;
                for (int j = 0; j < RAW_W; j++) {
                    int cx = ox + j;
                    if (cx < 0 || cx >= CANVAS_W) continue;

                    float wx = (j < margin) ? j / margin : (j > RAW_W - margin ? (RAW_W - j) / margin : 1.0f);
                    float wy = (i < margin) ? i / margin : (i > RAW_H - margin ? (RAW_H - i) / margin : 1.0f);
                    float w = std::max(wx * wy, 0.0001f);

                    canvas_accum.at<float>(cy, cx) += (float)curr_img.at<uchar>(i, j) * w;
                    canvas_weight.at<float>(cy, cx) += w;
                }
            }

            // 현재 타일을 다음 타일의 '왼쪽 타일'로 설정
            left_tile_img = curr_float;
        }

        // 메모리 관리
        if (y > 0) row_starter_images[y-1] = Mat();
        cout << "Row " << y << " processed." << endl;
    }

    // 3. 최종 결과 생성
    Mat final_result;
    divide(canvas_accum, canvas_weight, final_result);
    final_result.convertTo(final_result, CV_8U);

    imwrite("grid_stitched_result.png", final_result);
    return 0;
}
