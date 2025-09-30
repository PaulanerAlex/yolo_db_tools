import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.widgets import Button

def bb_picker(image, normalize=False, rotation_enabled=True):
    """
    Interactive picker for rotated rectangle on an image.

    Args:
        image (str or array-like): Path to image file or image array.
        normalize (bool): If True, returns coordinates and sizes normalized to [0,1] by image width/height.

    Returns:
        corners (list of (x, y)): The 4 rectangle corners in order [p0, p1, p2, p3].
            If normalize=True, coordinates are in [0,1], else in pixel coordinates.
        width (float): Length of side 1. Normalized if normalize=True.
        height (float): Length of side 2. Normalized if normalize=True.
        angle (float): Rotation angle in degrees from horizontal.
    """
    # Load image if filename given
    if isinstance(image, str):
        img = plt.imread(image)
    else:
        img = image

    img_h, img_w = img.shape[0], img.shape[1]

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img, origin='upper')  # top-left origin
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect('equal')

    # State variables
    angle = 0.0  # in degrees
    mode = 'init'  # init, drawing, edit
    p0 = None
    width = 0.0
    height = 0.0
    u_dir = np.array([1.0, 0.0])
    v_dir = np.array([0.0, 1.0])

    # Artists
    cross1 = Line2D([], [], lw=1, color='cyan')
    cross2 = Line2D([], [], lw=1, color='red')
    ax.add_line(cross1)
    ax.add_line(cross2)

    rect = Polygon([[0, 0]], closed=True, fill=False, lw=2, color='yellow')
    ax.add_patch(rect)

    handles = ax.scatter([], [], s=100, color='red', picker=5)

    # Finish Button
    ax_button = plt.axes([0.85, 0.92, 0.1, 0.05])
    btn = Button(ax_button, 'Fertig')

    dragging = None  # None, 'corner1', 'corner2'

    def update_cross(x, y):
        L = max(img_w, img_h)
        o = np.array([x, y])
        theta = np.deg2rad(angle)
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-np.sin(theta), np.cos(theta)])
        pu = np.vstack([o + u * -L, o + u * L])
        pv = np.vstack([o + v * -L, o + v * L])
        cross1.set_data(pu.T)
        cross2.set_data(pv.T)
        fig.canvas.draw_idle()

    def update_rectangle():
        p1 = p0 + u_dir * width
        p3 = p0 + v_dir * height
        p2 = p1 + v_dir * height
        rect.set_xy([p0, p1, p2, p3])
        handles.set_offsets([p1, p3])
        fig.canvas.draw_idle()

    def on_move(event):
        nonlocal width, height, angle
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        if mode == 'init':
            update_cross(x, y)
        elif mode == 'drawing':
            d = np.array([x, y]) - p0
            width = np.dot(d, u_dir)
            height = np.dot(d, v_dir)
            update_rectangle()
        elif mode == 'edit' and dragging:
            d = np.array([x, y]) - p0
            if dragging == 'corner1':
                length = np.hypot(*d)
                if length > 1e-3:
                    angle = np.rad2deg(np.arctan2(d[1], d[0]))
                    width = length
                    theta = np.deg2rad(angle)
                    u_dir[:] = [np.cos(theta), np.sin(theta)]
                    v_dir[:] = [-np.sin(theta), np.cos(theta)]
            elif dragging == 'corner2':
                length = np.hypot(*d)
                if length > 1e-3:
                    # derive angle so v_dir matches d
                    vx, vy = d / length
                    angle = np.rad2deg(np.arctan2(vy, -vx))
                    height = length
                    theta = np.deg2rad(angle)
                    u_dir[:] = [np.cos(theta), np.sin(theta)]
                    v_dir[:] = [-np.sin(theta), np.cos(theta)]
            update_rectangle()

    def on_scroll(event):
        nonlocal angle
        if mode != 'init' or event.inaxes != ax or event.xdata is None:
            return
        step = event.step if hasattr(event, 'step') else (1 if event.button == 'up' else -1)
        angle = (angle + 2 * step) % 360
        theta = np.deg2rad(angle)
        u_dir[:] = [np.cos(theta), np.sin(theta)]
        v_dir[:] = [-np.sin(theta), np.cos(theta)]
        update_cross(event.xdata, event.ydata)

    def on_click(event):
        nonlocal mode, p0
        if event.inaxes != ax or event.xdata is None:
            return
        if mode == 'init':
            p0 = np.array([event.xdata, event.ydata])
            mode = 'drawing'
            cross1.set_visible(False)
            cross2.set_visible(False)
        elif mode == 'drawing':
            mode = 'edit'

    def on_press(event):
        nonlocal dragging
        if mode == 'edit' and event.inaxes == ax and event.xdata is not None:
            pts = handles.get_offsets()
            dists = np.hypot(pts[:, 0] - event.xdata, pts[:, 1] - event.ydata)
            idx = dists.argmin()
            if dists[idx] < 10:
                dragging = 'corner1' if idx == 0 else 'corner2'

    def on_release(event):
        nonlocal dragging
        dragging = None

    def on_finish(event):
        plt.close(fig)

    # Event connections
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    if rotation_enabled == True:
        fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    btn.on_clicked(on_finish)

    plt.show()

    # Final geometry
    p1 = p0 + u_dir * width
    p3 = p0 + v_dir * height
    p2 = p1 + v_dir * height
    corners = [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]

    # Normalize if requested
    if normalize:
        corners = [(x / img_w, y / img_h) for x, y in corners]
        width /= img_w # FIXME: check, if width and height should be normalized
        height /= img_h

    return corners, width, height, angle

def show_bb(image, corners):
    """
    Displays the image with a bounding box drawn using the given normalized coordinates.

    Args:
        image (str or array-like): Path to image file or image array.
        corners (list of (x, y)): Normalized coordinates of the bounding box corners.
    """
    if isinstance(image, str):
        img = plt.imread(image)
    else:
        img = image

    img_h, img_w = img.shape[0], img.shape[1]
    # abs_corners = [(x * img_w, y * img_h) for x, y in corners]
    print(corners)

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img, origin='upper')
    poly = Polygon(corners, closed=True, fill=False, edgecolor='lime', lw=2)
    ax.add_patch(poly)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    data = bb_picker('test.png', normalize=True)

    # show_bb('test.png', data[0])

    print(data)
