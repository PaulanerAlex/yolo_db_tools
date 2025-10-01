import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.widgets import Button

def bb_picker_multiple(image, normalize=False, rotation_enabled=True):
    """
    Interactive picker for multiple rotated rectangles on an image.

    Args:
        image (str or array-like): Path to image file or image array.
        normalize (bool): If True, returns coordinates and sizes normalized to [0,1] by image width/height.
        rotation_enabled (bool): If True, allows rotation with scroll wheel.

    Returns:
        all_annotations (list): List of tuples, each containing (corners, width, height, angle)
            corners (list of (x, y)): The 4 rectangle corners in order [p0, p1, p2, p3].
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

    # State variables for current annotation
    angle = 0.0  # in degrees
    mode = 'init'  # init, drawing, edit
    p0 = None
    width = 0.0
    height = 0.0
    u_dir = np.array([1.0, 0.0])
    v_dir = np.array([0.0, 1.0])

    # Storage for completed annotations
    completed_annotations = []
    completed_polygons = []

    # Artists for current annotation
    cross1 = Line2D([], [], lw=1, color='cyan')
    cross2 = Line2D([], [], lw=1, color='red')
    ax.add_line(cross1)
    ax.add_line(cross2)

    rect = Polygon([[0, 0]], closed=True, fill=False, lw=2, color='yellow')
    ax.add_patch(rect)

    handles = ax.scatter([], [], s=100, color='red', picker=5)

    # Buttons
    ax_button_finish = plt.axes([0.85, 0.92, 0.1, 0.05])
    btn_finish = Button(ax_button_finish, 'Fertig')
    
    ax_button_add = plt.axes([0.85, 0.86, 0.1, 0.05])
    btn_add = Button(ax_button_add, 'Weitere Box')

    ax_button_undo = plt.axes([0.85, 0.80, 0.1, 0.05])
    btn_undo = Button(ax_button_undo, 'Rückgängig')

    dragging = None  # None, 'corner1', 'corner2'
    finished = False

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
        # Handle negative dimensions by adjusting the starting point and directions
        # This prevents horizontal mirroring when dragging upward
        
        # Determine actual corners based on width/height signs
        if width >= 0 and height >= 0:
            # Normal case: top-left to bottom-right
            corner0 = p0
            corner1 = p0 + u_dir * width
            corner3 = p0 + v_dir * height
            corner2 = corner1 + v_dir * height
        elif width < 0 and height >= 0:
            # Dragging left: adjust u_dir
            corner0 = p0 + u_dir * width  # Start from the left
            corner1 = p0
            corner3 = corner0 + v_dir * height
            corner2 = corner1 + v_dir * height
        elif width >= 0 and height < 0:
            # Dragging up: adjust v_dir
            corner0 = p0 + v_dir * height  # Start from above
            corner1 = corner0 + u_dir * width
            corner3 = p0
            corner2 = corner3 + u_dir * width
        else:
            # Both negative: dragging up and left
            corner0 = p0 + u_dir * width + v_dir * height
            corner1 = p0 + v_dir * height
            corner3 = p0 + u_dir * width
            corner2 = p0
        
        corners = [corner0, corner1, corner2, corner3]
        rect.set_xy(corners)
        handles.set_offsets([corner1, corner3])
        fig.canvas.draw_idle()

    def save_current_annotation():
        """Save the current annotation and reset for next one"""
        nonlocal mode, p0, angle, width, height
        if mode == 'edit' and p0 is not None:
            # Calculate correct corners based on width/height signs
            # This ensures consistent corner ordering regardless of drag direction
            if width >= 0 and height >= 0:
                # Normal case: top-left to bottom-right
                corner0 = p0
                corner1 = p0 + u_dir * width
                corner3 = p0 + v_dir * height
                corner2 = corner1 + v_dir * height
            elif width < 0 and height >= 0:
                # Dragging left: adjust u_dir
                corner0 = p0 + u_dir * width
                corner1 = p0
                corner3 = corner0 + v_dir * height
                corner2 = corner1 + v_dir * height
            elif width >= 0 and height < 0:
                # Dragging up: adjust v_dir
                corner0 = p0 + v_dir * height
                corner1 = corner0 + u_dir * width
                corner3 = p0
                corner2 = corner3 + u_dir * width
            else:
                # Both negative: dragging up and left
                corner0 = p0 + u_dir * width + v_dir * height
                corner1 = p0 + v_dir * height
                corner3 = p0 + u_dir * width
                corner2 = p0
            
            # Ensure consistent point ordering and positive dimensions
            corners = [tuple(corner0), tuple(corner1), tuple(corner2), tuple(corner3)]
            abs_width = abs(width)
            abs_height = abs(height)
            
            # Store the annotation with positive dimensions
            completed_annotations.append((corners, abs_width, abs_height, angle))
            
            # Create a permanent polygon for display
            if normalize:
                display_corners = [(x / img_w, y / img_h) for x, y in corners]
                display_corners = [(x * img_w, y * img_h) for x, y in display_corners]
            else:
                display_corners = corners
            
            completed_poly = Polygon(display_corners, closed=True, fill=False, lw=2, color='lime', alpha=0.7)
            ax.add_patch(completed_poly)
            completed_polygons.append(completed_poly)
            
            # Reset for next annotation
            reset_current_annotation()
            fig.canvas.draw_idle()

    def reset_current_annotation():
        """Reset the current annotation state"""
        nonlocal mode, p0, angle, width, height
        mode = 'init'
        p0 = None
        angle = 0.0
        width = 0.0
        height = 0.0
        u_dir[:] = [1.0, 0.0]
        v_dir[:] = [0.0, 1.0]
        
        # Hide current annotation elements
        cross1.set_visible(False)
        cross2.set_visible(False)
        rect.set_visible(False)
        handles.set_visible(False)

    def on_move(event):
        nonlocal width, height, angle
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        if mode == 'init':
            cross1.set_visible(True)
            cross2.set_visible(True)
            update_cross(x, y)
        elif mode == 'drawing':
            rect.set_visible(True)
            d = np.array([x, y]) - p0
            width = np.dot(d, u_dir)
            height = np.dot(d, v_dir)
            # Ensure we always display positive dimensions
            display_width = abs(width)
            display_height = abs(height)
            update_rectangle()
        elif mode == 'edit' and dragging:
            rect.set_visible(True)
            handles.set_visible(True)
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
        if not rotation_enabled:
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
            handles.set_visible(True)

    def on_press(event):
        nonlocal dragging
        if mode == 'edit' and event.inaxes == ax and event.xdata is not None:
            pts = handles.get_offsets()
            if len(pts) > 0:
                dists = np.hypot(pts[:, 0] - event.xdata, pts[:, 1] - event.ydata)
                idx = dists.argmin()
                if dists[idx] < 10:
                    dragging = 'corner1' if idx == 0 else 'corner2'

    def on_release(event):
        nonlocal dragging
        dragging = None

    def on_add_another(event):
        """Save current annotation and start a new one"""
        save_current_annotation()

    def on_undo(event):
        """Remove the last completed annotation"""
        if completed_annotations:
            completed_annotations.pop()
            if completed_polygons:
                poly = completed_polygons.pop()
                poly.remove()
                fig.canvas.draw_idle()

    def on_finish(event):
        nonlocal finished
        # Save current annotation if in edit mode
        if mode == 'edit' and p0 is not None:
            save_current_annotation()
        finished = True
        plt.close(fig)

    # Event connections
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    if rotation_enabled:
        fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    btn_finish.on_clicked(on_finish)
    btn_add.on_clicked(on_add_another)
    btn_undo.on_clicked(on_undo)

    # Initialize display
    reset_current_annotation()
    
    plt.show()

    # Normalize all annotations if requested
    if normalize:
        normalized_annotations = []
        for corners, w, h, a in completed_annotations:
            norm_corners = [(x / img_w, y / img_h) for x, y in corners]
            norm_w = w / img_w
            norm_h = h / img_h
            normalized_annotations.append((norm_corners, norm_w, norm_h, a))
        return normalized_annotations
    
    return completed_annotations

def bb_picker(image, normalize=True, rotation_enabled=True, multiple=False):
    """
    Interactive picker for rotated rectangle(s) on an image.

    Args:
        image (str or array-like): Path to image file or image array.
        normalize (bool): If True, returns coordinates and sizes normalized to [0,1] by image width/height.
        rotation_enabled (bool): If True, allows rotation with scroll wheel.
        multiple (bool): If True, allows multiple bounding boxes per image.

    Returns:
        If multiple=False (default):
            corners (list of (x, y)): The 4 rectangle corners in order [p0, p1, p2, p3].
            width (float): Length of side 1. Normalized if normalize=True.
            height (float): Length of side 2. Normalized if normalize=True.
            angle (float): Rotation angle in degrees from horizontal.
        
        If multiple=True:
            all_annotations (list): List of tuples, each containing (corners, width, height, angle)
    """
    if multiple:
        return bb_picker_multiple(image, normalize, rotation_enabled)
    
    # Original single bounding box implementation
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
        # Handle negative dimensions by adjusting the starting point and directions
        # This prevents horizontal mirroring when dragging upward
        
        # Determine actual corners based on width/height signs
        if width >= 0 and height >= 0:
            # Normal case: top-left to bottom-right
            corner0 = p0
            corner1 = p0 + u_dir * width
            corner3 = p0 + v_dir * height
            corner2 = corner1 + v_dir * height
        elif width < 0 and height >= 0:
            # Dragging left: adjust u_dir
            corner0 = p0 + u_dir * width  # Start from the left
            corner1 = p0
            corner3 = corner0 + v_dir * height
            corner2 = corner1 + v_dir * height
        elif width >= 0 and height < 0:
            # Dragging up: adjust v_dir
            corner0 = p0 + v_dir * height  # Start from above
            corner1 = corner0 + u_dir * width
            corner3 = p0
            corner2 = corner3 + u_dir * width
        else:
            # Both negative: dragging up and left
            corner0 = p0 + u_dir * width + v_dir * height
            corner1 = p0 + v_dir * height
            corner3 = p0 + u_dir * width
            corner2 = p0
        
        corners = [corner0, corner1, corner2, corner3]
        rect.set_xy(corners)
        handles.set_offsets([corner1, corner3])
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
        if not rotation_enabled:
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
            if len(pts) > 0:
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
    if rotation_enabled:
        fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    btn.on_clicked(on_finish)

    plt.show()

    # Final geometry - ensure positive dimensions and correct corner ordering
    # Calculate correct corners based on width/height signs
    if width >= 0 and height >= 0:
        # Normal case: top-left to bottom-right
        corner0 = p0
        corner1 = p0 + u_dir * width
        corner3 = p0 + v_dir * height
        corner2 = corner1 + v_dir * height
    elif width < 0 and height >= 0:
        # Dragging left: adjust u_dir
        corner0 = p0 + u_dir * width
        corner1 = p0
        corner3 = corner0 + v_dir * height
        corner2 = corner1 + v_dir * height
    elif width >= 0 and height < 0:
        # Dragging up: adjust v_dir
        corner0 = p0 + v_dir * height
        corner1 = corner0 + u_dir * width
        corner3 = p0
        corner2 = corner3 + u_dir * width
    else:
        # Both negative: dragging up and left
        corner0 = p0 + u_dir * width + v_dir * height
        corner1 = p0 + v_dir * height
        corner3 = p0 + u_dir * width
        corner2 = p0
    
    corners = [tuple(corner0), tuple(corner1), tuple(corner2), tuple(corner3)]
    abs_width = abs(width)
    abs_height = abs(height)

    # Normalize if requested
    if normalize:
        corners = [(x / img_w, y / img_h) for x, y in corners]
        abs_width /= img_w
        abs_height /= img_h

    return corners, abs_width, abs_height, angle

def show_bb(image, corners_list):
    """
    Displays the image with bounding box(es) drawn using the given normalized coordinates.

    Args:
        image (str or array-like): Path to image file or image array.
        corners_list (list): Either a single list of corners [(x, y), (x, y), ...] 
                            or a list of corner lists for multiple bounding boxes
    """
    if isinstance(image, str):
        img = plt.imread(image)
    else:
        img = image

    img_h, img_w = img.shape[0], img.shape[1]
    
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img, origin='upper')
    
    # Check if we have multiple bounding boxes or just one
    if len(corners_list) > 0 and isinstance(corners_list[0], tuple) and len(corners_list[0]) == 2:
        # Single bounding box: corners_list is [(x, y), (x, y), ...]
        print("Single bounding box:", corners_list)
        poly = Polygon(corners_list, closed=True, fill=False, edgecolor='lime', lw=2)
        ax.add_patch(poly)
    else:
        # Multiple bounding boxes: corners_list is [[(x, y), ...], [(x, y), ...], ...]
        print(f"Multiple bounding boxes: {len(corners_list)} boxes")
        colors = ['lime', 'red', 'blue', 'yellow', 'cyan', 'magenta']
        for i, corners in enumerate(corners_list):
            color = colors[i % len(colors)]
            poly = Polygon(corners, closed=True, fill=False, edgecolor=color, lw=2)
            ax.add_patch(poly)
    
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect('equal')
    plt.show()

def show_bb_from_annotations(image, annotations):
    """
    Displays the image with bounding boxes from annotation format.
    
    Args:
        image (str or array-like): Path to image file or image array.
        annotations (list): List of (corners, width, height, angle) tuples
    """
    corners_list = [corners for corners, _, _, _ in annotations]
    show_bb(image, corners_list)

if __name__ == "__main__":
    data = bb_picker('test.png', normalize=True)

    # show_bb('test.png', data[0])

    print(data)
