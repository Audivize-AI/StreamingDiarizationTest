#import <Foundation/Foundation.h>
#include "Uuid.hpp"

UUIDWrapper::UUIDWrapper() {
    NSUUID *u = [NSUUID UUID];
    uuid_t b;
    [u getUUIDBytes:b];
    std::memcpy(data, b, 16);
}