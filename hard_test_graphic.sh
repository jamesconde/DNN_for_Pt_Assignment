#!/bin/bash
lspci | grep -i --color 'vga\|3d\|2d'
