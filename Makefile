format:
	find ./src ./test -iname *.h -o -iname *.cc -o -iname *.cu | xargs clang-format -i
	find ./src ./test -iname *.h -o -iname *.cc -o -iname *.cu | xargs chmod 666