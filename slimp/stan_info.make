include makefile
cxxflags:
	@echo $(CXXFLAGS)
ldflags:
	@echo $(LDFLAGS) | sed 's/-Wl,-L,/-L/g'
libs:
	@echo $(LDLIBS)
