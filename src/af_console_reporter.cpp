
#include <arrayfire_benchmark.h>

namespace af {
  namespace benchmark {
    bool AFReporter::ReportContext(const Context& context) {
      ::benchmark::ConsoleReporter::ReportContext(context);
      char device_name[256];
      char platform[256];
      char toolkit[256];
      char compute[256];
      deviceInfo(device_name, platform, toolkit, compute);

      int major, minor, patch;
      af_get_version(&major, &minor, &patch);
      const char* revision = af_get_revision();

      printf("Context:\n"
              "ArrayFire v%d.%d.%d (%s) (Backend: %s Platform: %s)\n"
              "Device: %s (%s %s)\n",
              major, minor, patch, revision, platform, platform,
              device_name, compute, toolkit);
      return true;
    }

    AFReporter::AFReporter() :
      ::benchmark::ConsoleReporter() {
    }
  }
}
