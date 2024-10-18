#pragma once

#define DEBUG_ON

// 30 black, 31 red, 32 green, 33 yellow, 34 blue, 35 magenta, 36 cyan, 37 white

#ifdef DEBUG_ON
#define LOG_DEBUG(...)                                            \
    printf("[DEBUG][%s][%d][%s] ", __FILE__, __LINE__, __func__), \
        printf(__VA_ARGS__)
#define LOG_ERROR(...)                                                 \
    printf("\x1b[31m[ERROR][%s][%d][%s]\x1b[39m ", __FILE__, __LINE__, \
           __func__),                                                  \
        printf(__VA_ARGS__)
#define LOG_WARN(...)                                                 \
    printf("\x1b[33m[WARN][%s][%d][%s]\x1b[39m ", __FILE__, __LINE__, \
           __func__),                                                 \
        printf(__VA_ARGS__)
#define LOG_IMPORTANT(...)                                                 \
    printf("\x1b[32m[IMPORTANT][%s][%d][%s]\x1b[39m ", __FILE__, __LINE__, \
           __func__),                                                      \
        printf(__VA_ARGS__)
#define LOG_INFO(...)                                                 \
    printf("\x1b[36m[INFO][%s][%d][%s]\x1b[39m ", __FILE__, __LINE__, \
           __func__),                                                 \
        printf(__VA_ARGS__)
#else
#define LOG_DEBUG(...)
#endif
