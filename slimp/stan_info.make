STAN = $(CMDSTAN)/stan/
STAN_CPP_OPTIMS = 1
STAN_THREADS = 1
STAN_NO_RANGE_CHECKS = 1

include $(CMDSTAN)/makefile

cxxflags:
	@echo $(CPPFLAGS) $(CXXFLAGS)
ldflags:
	@echo $(LDFLAGS) | sed 's/-Wl,-L,/-L/g'
libs:
	@echo $(LDLIBS)
