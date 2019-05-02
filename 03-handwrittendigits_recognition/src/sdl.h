#ifndef _SRC_SDL_H
#define _SRC_SDL_H

#include <stdlib.h>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>


void init_sdl();

SDL_Surface* load_image(char *path);

SDL_Surface* display_image(SDL_Surface *img);

SDL_Surface *CreateSDLImage(int width, int height);

void wait_for_keypressed();


Uint32 get_pixel(SDL_Surface *surface, unsigned x, unsigned y);

void put_pixel(SDL_Surface *surface, unsigned x, unsigned y, Uint32 pixel);

void update_surface(SDL_Surface* screen, SDL_Surface* image);

void drawLineHor(SDL_Surface *img, unsigned int y, Uint32 color);

void drawLineVer(SDL_Surface *img, unsigned int x, Uint32 color);


#endif /* _SRC_SDL_H */
