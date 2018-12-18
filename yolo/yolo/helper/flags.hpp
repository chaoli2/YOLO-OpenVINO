#pragma once
#include <gflags/gflags.h>
#include <iostream>

namespace helper{

DEFINE_string(image, "", "input image");
DEFINE_string(m, "", "input inference graph");
DEFINE_string(w, "", "input inference weights");

bool ValidateName(const char* flag, const std::string& value){
    if(value.empty()){
        std::cout << "Please set input image. --image <IMG_PATH>" << std::endl;
        return false;
    }
    return true;
}

bool Validate_m(const char* flag, const std::string& value){
    if(value.empty()){
        std::cout << "Please set input inference graph. --m <.xml PATH>" << std::endl;
        return false;
    }
    return true;
}

bool Validate_w(const char* flag, const std::string& value){
    if(value.empty()){
        std::cout << "Please set input inference weights. --w <.bin PATH>" << std::endl;
        return false;
    }
    return true;
}

}