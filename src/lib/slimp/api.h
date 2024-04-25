#ifndef _9bc80a0a_760d_438a_8743_2cc4cb0c0b77
#define _9bc80a0a_760d_438a_8743_2cc4cb0c0b77

#if defined _WIN32
#  ifdef BUILDING_SLIMP
#    define SLIMP_API __declspec(dllexport)
#  else
#    define SLIMP_API __declspec(dllimport)
#  endif
#else
#  define SLIMP_API __attribute__ ((visibility ("default")))
#endif

#endif // _9bc80a0a_760d_438a_8743_2cc4cb0c0b77
