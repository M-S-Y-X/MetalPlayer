//
//  HomographyEngine.m
//  纯 C 算法实现，无 C++ 依赖
//

#import "HomographyEngine.h"
#import <stdlib.h>
#import <string.h>
#import <math.h>
#import <time.h>

@implementation HomographyResult
@end

// ============================================================
//  纯 C 算法部分
// ============================================================

// ---------- 灰度提取 ----------
static unsigned char* getGrayFromCGImage(CGImageRef cg, int* outW, int* outH) {
    size_t w = CGImageGetWidth(cg);
    size_t h = CGImageGetHeight(cg);
    *outW = (int)w; *outH = (int)h;
    CGColorSpaceRef graySpace = CGColorSpaceCreateDeviceGray();
    CGContextRef ctx = CGBitmapContextCreate(NULL, w, h, 8, w, graySpace, kCGImageAlphaNone);
    CGContextDrawImage(ctx, CGRectMake(0,0,w,h), cg);
    unsigned char* data = (unsigned char*)CGBitmapContextGetData(ctx);
    size_t len = w * h;
    unsigned char* copy = (unsigned char*)malloc(len);
    memcpy(copy, data, len);
    CGContextRelease(ctx);
    CGColorSpaceRelease(graySpace);
    return copy;
}

// ---------- FAST 角点 ----------
typedef struct { int x; int y; } Corner;

static int fast_corners(const unsigned char* gray, int w, int h, int threshold, Corner** outCorners) {
    const int offsets[16][2] = {
        {0,-3},{1,-3},{2,-2},{3,-1},{3,0},{3,1},{2,2},{1,3},
        {0,3},{-1,3},{-2,2},{-3,1},{-3,0},{-3,-1},{-2,-2},{-1,-3}
    };
    Corner* corners = NULL;
    int count = 0, capacity = 0;
    for (int y=3; y<h-3; ++y) {
        for (int x=3; x<w-3; ++x) {
            int center = gray[y*w + x];
            int v1 = gray[(y+offsets[0][1])*w + (x+offsets[0][0])];
            int v5 = gray[(y+offsets[4][1])*w + (x+offsets[4][0])];
            int v9 = gray[(y+offsets[8][1])*w + (x+offsets[8][0])];
            int v13= gray[(y+offsets[12][1])*w + (x+offsets[12][0])];
            int brighter = (v1 - center > threshold) + (v5 - center > threshold) +
                           (v9 - center > threshold) + (v13 - center > threshold);
            int darker   = (center - v1 > threshold) + (center - v5 > threshold) +
                           (center - v9 > threshold) + (center - v13 > threshold);
            if (brighter < 3 && darker < 3) continue;
            int bright_cnt=0, dark_cnt=0;
            for (int i=0; i<16; ++i) {
                int val = gray[(y+offsets[i][1])*w + (x+offsets[i][0])];
                if (val - center > threshold) bright_cnt++;
                else if (center - val > threshold) dark_cnt++;
            }
            if (bright_cnt >= 12 || dark_cnt >= 12) {
                if (count >= capacity) {
                    capacity = capacity ? capacity*2 : 1024;
                    corners = (Corner*)realloc(corners, capacity * sizeof(Corner));
                }
                corners[count].x = x;
                corners[count].y = y;
                count++;
            }
        }
    }
    *outCorners = corners;
    return count;
}

// ---------- BRIEF ----------
static int brief_pattern[256][2];

static void init_brief_pattern(void) {
    srand(12345);
    for (int i=0; i<256; ++i) {
        brief_pattern[i][0] = (rand() % 31) - 15;
        brief_pattern[i][1] = (rand() % 31) - 15;
    }
}

static void compute_brief(const unsigned char* gray, int w, int h, int x, int y, unsigned char desc[32]) {
    memset(desc, 0, 32);
    for (int i=0; i<256; ++i) {
        int dx = brief_pattern[i][0], dy = brief_pattern[i][1];
        int x1 = x + dx; if (x1 < 0) x1=0; if (x1 >= w) x1=w-1;
        int y1 = y + dy; if (y1 < 0) y1=0; if (y1 >= h) y1=h-1;
        int x2 = x - dx; if (x2 < 0) x2=0; if (x2 >= w) x2=w-1;
        int y2 = y - dy; if (y2 < 0) y2=0; if (y2 >= h) y2=h-1;
        unsigned char bit = (gray[y1*w + x1] < gray[y2*w + x2]) ? 1 : 0;
        desc[i >> 3] |= (bit << (i & 7));
    }
}

static int hamming_distance(const unsigned char a[32], const unsigned char b[32]) {
    int d=0;
    for (int i=0; i<32; ++i) {
        unsigned char x = a[i] ^ b[i];
        while (x) { ++d; x &= x-1; }
    }
    return d;
}

typedef struct { int idxA; int idxB; int dist; } Match;

static int match_features(const unsigned char** descA, int countA,
                          const unsigned char** descB, int countB,
                          Match** outMatches) {
    Match* matches = NULL;
    int mcount = 0, mcap = 0;
    for (int i=0; i<countB; ++i) {
        int best=9999, second=9999, bestIdx=-1;
        for (int j=0; j<countA; ++j) {
            int d = hamming_distance(descB[i], descA[j]);
            if (d < best) { second=best; best=d; bestIdx=j; }
            else if (d < second) second = d;
        }
        if (bestIdx != -1 && (float)best / second < 0.8f) {
            if (mcount >= mcap) {
                mcap = mcap ? mcap*2 : 256;
                matches = (Match*)realloc(matches, mcap * sizeof(Match));
            }
            matches[mcount].idxA = bestIdx;
            matches[mcount].idxB = i;
            matches[mcount].dist = best;
            mcount++;
        }
    }
    *outMatches = matches;
    return mcount;
}

// ---------- 4点 DLT 高斯消元 ----------
static bool solve_homography_4points(const float src[4][2], const float dst[4][2], float H[3][3]) {
    float A[8][8], b[8];
    for (int i=0; i<4; ++i) {
        float x=src[i][0], y=src[i][1], u=dst[i][0], v=dst[i][1];
        A[2*i][0]=x; A[2*i][1]=y; A[2*i][2]=1;
        A[2*i][3]=0; A[2*i][4]=0; A[2*i][5]=0;
        A[2*i][6]=-u*x; A[2*i][7]=-u*y;
        b[2*i]=u;
        A[2*i+1][0]=0; A[2*i+1][1]=0; A[2*i+1][2]=0;
        A[2*i+1][3]=x; A[2*i+1][4]=y; A[2*i+1][5]=1;
        A[2*i+1][6]=-v*x; A[2*i+1][7]=-v*y;
        b[2*i+1]=v;
    }
    float Aug[8][9];
    for (int i=0; i<8; ++i) {
        for (int j=0; j<8; ++j) Aug[i][j] = A[i][j];
        Aug[i][8] = b[i];
    }
    for (int col=0; col<8; ++col) {
        int maxRow=col; float maxVal=fabs(Aug[col][col]);
        for (int row=col+1; row<8; ++row) {
            float v = fabs(Aug[row][col]);
            if (v > maxVal) { maxVal=v; maxRow=row; }
        }
        if (maxVal < 1e-12) return false;
        for (int j=col; j<=8; ++j) {
            float tmp = Aug[col][j];
            Aug[col][j] = Aug[maxRow][j];
            Aug[maxRow][j] = tmp;
        }
        float pivot = Aug[col][col];
        for (int j=col; j<=8; ++j) Aug[col][j] /= pivot;
        for (int row=0; row<8; ++row) {
            if (row == col) continue;
            float factor = Aug[row][col];
            for (int j=col; j<=8; ++j) Aug[row][j] -= factor * Aug[col][j];
        }
    }
    float h[8];
    for (int i=0; i<8; ++i) h[i] = Aug[i][8];
    H[0][0]=h[0]; H[0][1]=h[1]; H[0][2]=h[2];
    H[1][0]=h[3]; H[1][1]=h[4]; H[1][2]=h[5];
    H[2][0]=h[6]; H[2][1]=h[7]; H[2][2]=1.0f;
    return true;
}

// ---------- RANSAC ----------
static bool ransac_homography(const float* srcX, const float* srcY,
                              const float* dstX, const float* dstY,
                              int n, float H[3][3],
                              int iterations, float inlier_thresh) {
    if (n < 4) return false;
    int bestInlierCnt=0;
    float bestH[3][3];
    for (int iter=0; iter<iterations; ++iter) {
        int idx[4];
        for (int i=0; i<4; ++i) idx[i] = rand() % n;
        float src[4][2], dst[4][2];
        for (int i=0; i<4; ++i) {
            src[i][0] = srcX[idx[i]]; src[i][1] = srcY[idx[i]];
            dst[i][0] = dstX[idx[i]]; dst[i][1] = dstY[idx[i]];
        }
        float Hcur[3][3];
        if (!solve_homography_4points(src, dst, Hcur)) continue;
        int inlierCnt=0;
        for (int i=0; i<n; ++i) {
            float x=srcX[i], y=srcY[i];
            float denom = Hcur[2][0]*x + Hcur[2][1]*y + Hcur[2][2];
            if (fabs(denom) < 1e-8) continue;
            float u_pred = (Hcur[0][0]*x + Hcur[0][1]*y + Hcur[0][2]) / denom;
            float v_pred = (Hcur[1][0]*x + Hcur[1][1]*y + Hcur[1][2]) / denom;
            float err = (u_pred - dstX[i])*(u_pred - dstX[i]) +
                        (v_pred - dstY[i])*(v_pred - dstY[i]);
            if (err < inlier_thresh*inlier_thresh) inlierCnt++;
        }
        if (inlierCnt > bestInlierCnt) {
            bestInlierCnt = inlierCnt;
            memcpy(bestH, Hcur, sizeof(bestH));
        }
    }
    if (bestInlierCnt < 8) return false;
    memcpy(H, bestH, sizeof(bestH));
    return true;
}

// ---------- 画布参数 ----------
static void compute_canvas_params(int wA, int hA, int wB, int hB,
                                  const float H[3][3],
                                  int* outW, int* outH,
                                  int* outX, int* outY) {
    float cornersB[4][2] = {{0,0}, {(float)wB,0}, {(float)wB,(float)hB}, {0,(float)hB}};
    float warpCorners[4][2];
    for (int i=0; i<4; ++i) {
        float x = cornersB[i][0], y = cornersB[i][1];
        float denom = H[2][0]*x + H[2][1]*y + H[2][2];
        warpCorners[i][0] = (H[0][0]*x + H[0][1]*y + H[0][2]) / denom;
        warpCorners[i][1] = (H[1][0]*x + H[1][1]*y + H[1][2]) / denom;
    }
    float allX[8] = {0, (float)wA, (float)wA, 0,
                     warpCorners[0][0], warpCorners[1][0], warpCorners[2][0], warpCorners[3][0]};
    float allY[8] = {0, 0, (float)hA, (float)hA,
                     warpCorners[0][1], warpCorners[1][1], warpCorners[2][1], warpCorners[3][1]};
    float xMin = allX[0], xMax = allX[0];
    float yMin = allY[0], yMax = allY[0];
    for (int i=1; i<8; ++i) {
        if (allX[i] < xMin) xMin = allX[i];
        if (allX[i] > xMax) xMax = allX[i];
        if (allY[i] < yMin) yMin = allY[i];
        if (allY[i] > yMax) yMax = allY[i];
    }
    *outX = (int)floor(xMin);
    *outY = (int)floor(yMin);
    *outW = (int)ceil(xMax - xMin);
    *outH = (int)ceil(yMax - yMin);
}

// ============================================================
//  Objective-C 实现
// ============================================================
@implementation HomographyEngine

+ (HomographyResult *)computeHomographyFromImageA:(CGImageRef)imageA
                                            imageB:(CGImageRef)imageB {
    HomographyResult *result = [[HomographyResult alloc] init];
    result.success = NO;
    if (!imageA || !imageB) return result;
    
    int wA, hA, wB, hB;
    unsigned char* grayA = getGrayFromCGImage(imageA, &wA, &hA);
    unsigned char* grayB = getGrayFromCGImage(imageB, &wB, &hB);
    if (!grayA || !grayB) {
        if (grayA) free(grayA);
        if (grayB) free(grayB);
        return result;
    }
    
    Corner* cornersA = NULL;
    Corner* cornersB = NULL;
    int cntA = fast_corners(grayA, wA, hA, 25, &cornersA);
    int cntB = fast_corners(grayB, wB, hB, 25, &cornersB);
    if (cntA < 8 || cntB < 8) {
        free(grayA); free(grayB);
        if (cornersA) free(cornersA);
        if (cornersB) free(cornersB);
        return result;
    }
    
    init_brief_pattern();
    unsigned char** descA = (unsigned char**)malloc(cntA * sizeof(unsigned char*));
    unsigned char** descB = (unsigned char**)malloc(cntB * sizeof(unsigned char*));
    for (int i=0; i<cntA; ++i) {
        descA[i] = (unsigned char*)malloc(32);
        compute_brief(grayA, wA, hA, cornersA[i].x, cornersA[i].y, descA[i]);
    }
    for (int i=0; i<cntB; ++i) {
        descB[i] = (unsigned char*)malloc(32);
        compute_brief(grayB, wB, hB, cornersB[i].x, cornersB[i].y, descB[i]);
    }
    free(grayA); free(grayB);
    
    Match* matches = NULL;
    int mcnt = match_features((const unsigned char**)descA, cntA,
                              (const unsigned char**)descB, cntB,
                              &matches);
    if (mcnt < 8) {
        for (int i=0; i<cntA; ++i) free(descA[i]);
        for (int i=0; i<cntB; ++i) free(descB[i]);
        free(descA); free(descB);
        if (cornersA) free(cornersA);
        if (cornersB) free(cornersB);
        if (matches) free(matches);
        return result;
    }
    
    float* srcX = (float*)malloc(mcnt * sizeof(float));
    float* srcY = (float*)malloc(mcnt * sizeof(float));
    float* dstX = (float*)malloc(mcnt * sizeof(float));
    float* dstY = (float*)malloc(mcnt * sizeof(float));
    for (int i=0; i<mcnt; ++i) {
        int idxA = matches[i].idxA;
        int idxB = matches[i].idxB;
        srcX[i] = (float)cornersB[idxB].x;
        srcY[i] = (float)cornersB[idxB].y;
        dstX[i] = (float)cornersA[idxA].x;
        dstY[i] = (float)cornersA[idxA].y;
    }
    free(matches);
    
    srand((unsigned)time(NULL));
    float H[3][3];
    bool ok = ransac_homography(srcX, srcY, dstX, dstY, mcnt, H, 3000, 5.0f);
    free(srcX); free(srcY); free(dstX); free(dstY);
    for (int i=0; i<cntA; ++i) free(descA[i]);
    for (int i=0; i<cntB; ++i) free(descB[i]);
    free(descA); free(descB);
    free(cornersA); free(cornersB);
    
    if (!ok) return result;
    
    int canvasW, canvasH, offX, offY;
    compute_canvas_params(wA, hA, wB, hB, H, &canvasW, &canvasH, &offX, &offY);
    
    float T[3][3] = {{1,0,(float)-offX}, {0,1,(float)-offY}, {0,0,1}};
    float H_new[3][3];
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j) {
            H_new[i][j] = 0;
            for (int k=0; k<3; ++k) H_new[i][j] += T[i][k] * H[k][j];
        }
    
    simd_float3x3 H_metal;
    for (int r=0; r<3; ++r)
        for (int c=0; c<3; ++c)
            H_metal.columns[c][r] = H_new[r][c];
    
    result.H = H_metal;
    result.canvasWidth = canvasW;
    result.canvasHeight = canvasH;
    result.offsetX = offX;
    result.offsetY = offY;
    result.success = YES;
    return result;
}

@end
