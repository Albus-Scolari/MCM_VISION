try:
    from PIL import Image
except ImportError:
    import Image



background = Image.open("C:\\Users\\alberto.scolari\\source\\repos\\humans-RME\\DATA\\scaled_image_0.png")
overlay = Image.open("C:\\Users\\alberto.scolari\\source\\repos\\humans-RME\\DATA\\stitched_r.png")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.5)
new_img.save("new.png","PNG")