import os
import http.server
import socketserver

PORT = 8000

class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    SimpleHTTPServer that supports HTTP range requests, allowing
    HTML5 video/audio elements to scrub/seek properly.
    """
    def send_head(self):
        if 'Range' not in self.headers:
            self.range = None
            return super().send_head()
        
        try:
            self.range = self.parse_range_header(self.headers['Range'])
        except ValueError:
            self.send_error(http.HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
            
        path = self.translate_path(self.path)
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(http.HTTPStatus.NOT_FOUND, "File not found")
            return None

        fs = os.fstat(f.fileno())
        file_len = fs[6]
        
        start, end = self.range
        if start is None:
            start = 0
        if end is None or end >= file_len:
            end = file_len - 1
            
        self.send_response(http.HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_len}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        
        f.seek(start)
        self.range_length = end - start + 1
        return f

    def parse_range_header(self, range_header):
        if not range_header.startswith('bytes='):
            raise ValueError
        parts = range_header[6:].split('-')
        if len(parts) != 2:
            raise ValueError
            
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        return start, end

    def copyfile(self, source, outputfile):
        try:
            if not hasattr(self, 'range'):
                return super().copyfile(source, outputfile)
                
            if self.range is None:
                return super().copyfile(source, outputfile)
                
            chunk_size = 65536
            remains = self.range_length
            while remains > 0:
                chunk = source.read(min(chunk_size, remains))
                if not chunk:
                    break
                outputfile.write(chunk)
                remains -= len(chunk)
        except (ConnectionError, BrokenPipeError):
            # This cleanly catches WinError 10054 and WinError 10053!
            # The browser just decided it had buffered enough of the video.
            pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), RangeRequestHandler) as httpd:
    print(f"Serving at http://localhost:{PORT} with Range requests support...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
