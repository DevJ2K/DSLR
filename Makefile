########################################
########## VARIABLES
FILES_DIR =

########################################
########## COLORS
DEF_COLOR = \033[0;39m
GRAY = \033[1;90m
RED = \033[1;91m
GREEN = \033[1;92m
YELLOW = \033[1;93m
BLUE = \033[1;94m
MAGENTA = \033[1;95m
CYAN = \033[1;96m
WHITE = \033[1;97m

########################################
########## RULES

all: test

test:
		@echo "$(GREEN)########################################"
		@echo "$(GREEN)######## PYTEST $(DEF_COLOR)"
		@rm -r tests/out
		@mkdir tests/out
		@python3 -m pytest tests/*.py -v && echo "$(GREEN)SUCCESS $(DEF_COLOR)" || echo "$(RED)FAILURE $(DEF_COLOR)"


clean:
		rm -rf __pycache__ tests/__pycache__


.PHONY: all test clean
