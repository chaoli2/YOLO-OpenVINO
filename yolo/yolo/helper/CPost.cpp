#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
#include <tuple>
#include <map>
#include <numeric>
#include <algorithm>

// http://www.cnblogs.com/hustxujinkang/p/4614437.html
template<typename T>
std::vector<int> argsort_descend(const std::vector<T>& v){
    int Len = v.size();
    std::vector<int> idx(Len, 0);
    for(int i = 0; i < Len; i++){
          idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(), [&v](int a, int b){return v[a] > v[b];});
    return idx;
}


extern "C" void yolov1(float * predictions, int32_t S, int32_t N, int32_t C,
                        float threshNMS, float objectnessThresh,
                        float *dets, int32_t * pCnt)
{
	//float top2Thr = thresh * 4 / 3;	// Top2 score threshold

    float * prob       = predictions;                   //SxSxC
    float * objectness = predictions + S * S * C;       //SxSxN
    float * loc        = predictions + S * S * (C + N); //SxSxNx4

    //here index i is 1-D scan order of SxSxN predictions
    //     index l is 1-D scan order of SxS location
    std::vector<float> all_x0(S*S*N, 0);
    std::vector<float> all_y0(S*S*N, 0);
    std::vector<float> all_x1(S*S*N, 0);
    std::vector<float> all_y1(S*S*N, 0);
    std::vector<std::vector<float> > all_p(C,  std::vector<float>(S*S*N, 0) );

#define X(i) loc[i*4 + 0]
#define Y(i) loc[i*4 + 1]
#define W(i) loc[i*4 + 2]
#define H(i) loc[i*4 + 3]

    //Collect all data
    int x, y, n, c;
    int l=0, i=0;
	for(l=i=y=0; y<S; y++)
    for(x=0; x<S; x++, l++)
    for (n=0; n<N; ++n, i++) {
        //fix anchor location
        float w =  W(i)*W(i);
        float h =  H(i)*H(i);
        all_x0[i] = (X(i) + x)*(1.0/S) - w*0.5f;
        all_y0[i] = (Y(i) + y)*(1.0/S) - h*0.5f;
        all_x1[i] = all_x0[i] + w;
        all_y1[i] = all_y0[i] + h;
        for (c=0; c<C; c++){
            float pc = objectness[i]*prob[l*C + c];
            all_p[c][i] = pc > objectnessThresh ? pc:0;
        }
    }

    // NMS, at same location (IOU > fNMS), suppress smaller probs along same class,
    // This is a key step for high precision because multiple nearby predictions exists for single object
    // We need suppress redundant detection to lower false positive
    //     but what if two objects of same class occupy same locations ?
    //     for now, we can only trade off these two situations by IOU threshold
    for (c=0; c<C; c++){
        std::vector<int> idx = argsort_descend<float>(all_p[c]);

        for(int i0 = 0; i0 < idx.size(); i0++){
            auto i = idx[i0];

            if(all_p[c][i] == 0) continue;

            float si = (all_x1[i] - all_x0[i]) * (all_y1[i] - all_y0[i]);

            for(int j0=i0+1;j0<idx.size();j0++){
                auto j = idx[j0];
                float mx0 = std::max(all_x0[i], all_x0[j]);
                float my0 = std::max(all_y0[i], all_y0[j]);
                float mx1 = std::min(all_x1[i], all_x1[j]);
                float my1 = std::min(all_y1[i], all_y1[j]);
                float ovx = std::max(0.f, mx1 -mx0);
                float ovy = std::max(0.f, my1 -my0);
                float inters = ovx * ovy;

                float sj = (all_x1[j] - all_x0[j]) * (all_y1[j] - all_y0[j]);
                float uni = si + sj - inters;
                if( inters/uni >= threshNMS )
                    all_p[c][j] = 0;
            }
        }
    }

    //now collect all result boxes
    // dets: (left,top,right,bottom,classid,confident)
    int k = 0;
    for(i=0;i<S*S*N;i++){
        for (c=0; c<C; c++){
            if (all_p[c][i] == 0) continue;
            if(k < *pCnt){
                dets[k*6 + 0] = std::min(std::max(all_x0[i], 0.0f), 1.0f);
                dets[k*6 + 1] = std::min(std::max(all_y0[i], 0.0f), 1.0f);
                dets[k*6 + 2] = std::min(std::max(all_x1[i], 0.0f), 1.0f);
                dets[k*6 + 3] = std::min(std::max(all_y1[i], 0.0f), 1.0f);
                dets[k*6 + 4] = c + 1;
                dets[k*6 + 5] = all_p[c][i];
            }
            k++;
        }
    }

    *pCnt = k;
}

//===================================================================================================================================================

typedef struct {
    float x, y, w, h;
} box;

typedef struct {
    int index;
    int cclass;
    float **probs;
} sortable_bbox;

float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

float box_union(box a, box b) {
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(box a, box b) {
    return box_intersection(a, b) / box_union(a, b);
}

int nms_comparator(const void *pa, const void *pb) {
    sortable_bbox a = *reinterpret_cast<const sortable_bbox *>(pa);
    sortable_bbox b = *reinterpret_cast<const sortable_bbox *>(pb);
    float diff = a.probs[a.index][b.cclass] - b.probs[b.index][b.cclass];
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

int max_index(float *a, int n) {
    if (n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max) {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

void get_detection_boxes(float *predictions, float thresh, float **probs, box *boxes, int classes) {
    int ln = 2;
    int side = 7;

    int i, j, n;
    for (i = 0; i < side * side; ++i) {
        int row = i / side;
        int col = i % side;
        for (n = 0; n < ln; ++n) {
            int index = i * ln + n;
            int p_index = side * side * classes + i * ln + n;
            double scale = predictions[p_index];
            int box_index = side * side * (classes + ln) + (i * ln + n) * 4;

            boxes[index].x = (predictions[box_index + 0] + col) / side;
            boxes[index].y = (predictions[box_index + 1] + row) / side;
            boxes[index].w = pow(predictions[box_index + 2], 2);
            boxes[index].h = pow(predictions[box_index + 3], 2);
            for (j = 0; j < classes; ++j) {
                int class_index = i * classes;
                double prob = scale * predictions[class_index + j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
        }
    }
}

void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh) {
    int i, j, k;
    sortable_bbox *s = (sortable_bbox *) calloc(total, sizeof(sortable_bbox));

    for (i = 0; i < total; ++i) {
        s[i].index = i;
        s[i].cclass = 0;
        s[i].probs = probs;
    }

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            s[i].cclass = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for (i = 0; i < total; ++i) {
            if (probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for (j = i + 1; j < total; ++j) {
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh) {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }

    free(s);
}

std::vector<std::vector<float> > print_detections(int imw, int imh, int num, float thresh, box *boxes, float **probs, int classes) {
    std::vector<std::vector<float> > ret;
    int i;
    for (i = 0; i < num; ++i) {
        int idxClass = max_index(probs[i], classes);
        float prob = probs[i][idxClass];

        if (prob > thresh) {
            box b = boxes[i];
            int left = (b.x - b.w / 2.) * imw;
            int right = (b.x + b.w / 2.) * imw;
            int top = (b.y - b.h / 2.) * imh;
            int bot = (b.y + b.h / 2.) * imh;

            if (left < 0) left = 0;
            if (right > imw - 1) right = imw - 1;
            if (top < 0) top = 0;
            if (bot > imh - 1) bot = imh - 1;

            if(1){
                //dets: (left,top,right,bottom,classid,confident)
                std::vector<float> v(6);
                v[0] = b.x - b.w / 2.;
                v[1] = b.y - b.h / 2.;
                v[2] = b.x + b.w / 2.;
                v[3] = b.y + b.h / 2.;
                v[4] = idxClass;
                v[5] = prob;
                ret.push_back(v);
            }else{
                std::string label;
                label = std::to_string(idxClass);
                label = "label " +label;
                printf("%s: %.0f%% ", label.c_str(), prob * 100);
                printf("[(%d %d), (%d %d)]\n", left, top, right, bot);
            }
        }
    }
    return ret;
}

int entry_index(int lw, int lh, int lcoords, int lclasses, int lnum, int batch, int location, int entry) {
    int n =   location / (lw * lh);
    int loc = location % (lw * lh);
    int loutputs = lh * lw * lnum * (lclasses + lcoords + 1);
    return batch* loutputs + n * lw * lh * (lcoords + lclasses + 1) + entry * lw * lh + loc;
}

box get_region_box(float *x, const float *biases, int n, int index, int i, int j, int w, int h, int stride) {
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;

    return b;
}

void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative) {
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}

void get_region_boxes(float *predictions, int lw, int lh, int lcoords, int lclasses, int lnum,
    int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, int relative, const float *anchors) {
    int i,j,n;

    for (i = 0; i < lw * lh; ++i){
        int row = i / lw;
        int col = i % lw;
        for(n = 0; n < lnum; ++n){
            int index = n * lw * lh + i;
            for(j = 0; j < lclasses; ++j){
                probs[index][j] = 0;
            }
            int obj_index = entry_index(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords);
            int box_index = entry_index(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, 0);
            float scale = predictions[obj_index];

            boxes[index] = get_region_box(predictions, anchors, n, box_index, col, row, lw, lh, lw * lh);

            float max = 0;
            for(j = 0; j < lclasses; ++j){
                int class_index = entry_index(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords + 1 + j);
                float prob = scale * predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                if(prob > max) max = prob;
            }
            probs[index][lclasses] = max;

        }
    }
    correct_region_boxes(boxes, lw * lh * lnum, w, h, netw, neth, relative);
}


std::vector<std::vector<float> > yolov2_postprocess(float *data, int size, const float *anchors, float thresh, float nms, int classes) {

    std::vector<std::vector<float> > ret;
    int coords = 4;
    int num = 5;

    int imw = 416;
    int imh = 416;

    int lw = 13;
    int lh = 13;

    box *boxes = (box *) malloc(lw * lh * num * sizeof(box));
    float **probs = (float **) malloc(lw * lh * num * sizeof(float *));
    for (int j = 0; j < lw * lh * num ; ++j)
        probs[j] = (float *) malloc((classes + 1) * sizeof(float));

    get_region_boxes(data, lw, lh, coords, classes, num, imw, imh, imw, imh, thresh, probs, boxes, 1, anchors);
    do_nms_sort(boxes, probs, lw * lh * num, classes, nms);
    ret = print_detections(imw, imh, lw * lh * num, thresh, boxes, probs, classes);

    for (int j = 0; j < lw * lh * num ; ++j)
        free(probs[j]);
    free(probs);
    free(boxes);
    
    return ret;
}


// dets: (left,top,right,bottom,classid,confident)
extern "C" void yolov2(float * predictions, int32_t size, int classes,
                        float threshNMS, float objectnessThresh,
                        float *dets, int32_t * pCnt)
{
    const float TINY_YOLOV2_ANCHORS[] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};
    std::vector<std::vector<float> > ret;
    
    ret = yolov2_postprocess(predictions, size, TINY_YOLOV2_ANCHORS, objectnessThresh, threshNMS, classes);
    
    //now collect all result boxes
    // dets: (left,top,right,bottom,classid,confident)

    for(int k=0;k<ret.size() && k<*pCnt;k++){
        dets[k*6 + 0] = std::min(std::max(ret[k][0], 0.0f), 1.0f);
        dets[k*6 + 1] = std::min(std::max(ret[k][1], 0.0f), 1.0f);
        dets[k*6 + 2] = std::min(std::max(ret[k][2], 0.0f), 1.0f);
        dets[k*6 + 3] = std::min(std::max(ret[k][3], 0.0f), 1.0f);
        dets[k*6 + 4] = ret[k][4] + 1;
        dets[k*6 + 5] = ret[k][5];
    }
    *pCnt = ret.size();
}


