#import <Foundation/Foundation.h>
#include "Uuid.h"

UUID::UUID() {
    NSUUID *u = [NSUUID UUID];
    uuid_t b;
    [u getUUIDBytes:b];
    std::memcpy(data, b, 16);
}