package com.example.alexnet;



import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;
import com.google.android.material.navigation.NavigationView;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.provider.DocumentsContract;
import android.util.Log;
import android.view.View;
import android.net.Uri;
import android.view.MenuItem;
import android.os.Bundle;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.view.GravityCompat;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

import android.view.Menu;
import android.widget.Toast;
import android.provider.MediaStore;
import android.database.Cursor;
import android.Manifest;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;

import static java.lang.System.in;


public class MainActivity extends AppCompatActivity
        implements NavigationView.OnNavigationItemSelectedListener , ClassifyFragment.OnFragmentInteractionListener{

    private ClassifyFragment clsfragments;
    FragmentManager fm;

    private static final int FILE_SELECT_CODE = 520;
    private static final int GET_PERMISSION = 520;
    private String image_path="";

    private int target_size=227;
    private float[] mean={104.f, 117.f, 123.f};
    private String model="ncnn_alexnet.bin";
    private String param="ncnn_alexnet.param";
    private String labels="imagenet1000_preprocessed_labels.txt";
    private netlib net =new netlib();
    private long net_env=0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        FloatingActionButton fab = findViewById(R.id.fab);


        try{
            fab.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        Snackbar.make(view, "cv classification" , Snackbar.LENGTH_LONG)
                                .setAction("Action", null).show();
                    }});

        DrawerLayout drawer = findViewById(R.id.drawer_layout);
        NavigationView navigationView = findViewById(R.id.nav_view);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(
                this, drawer, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawer.addDrawerListener(toggle);
        toggle.syncState();
        navigationView.setNavigationItemSelectedListener(this);
        //add by clause
        fm= getSupportFragmentManager();
        navigationView.setCheckedItem(R.id.nav_home);
        }catch (Exception e){
            e.printStackTrace();
        }

        //ncnn init
        //异步加载
        final Activity act=this;
        new Thread() {
            public void run() {
                // 1.执行耗时操作

                        String model_path=getAssetsCacheFile(act,model);
                        String param_path=getAssetsCacheFile(act,param);
                        String label_path=getAssetsCacheFile(act,labels);
                        Log.d("asserts path", "model path:"+model_path);
                        Log.d("asserts path", "param path:"+param_path);
                        Log.d("asserts path", "label path:"+label_path);
                        net_env=net.initEnv(model_path,param_path,label_path,mean,3,target_size);
                        //清除缓存
                        File mofile=new File(model_path);
                        if(mofile.exists()){
                            Log.d("remove model data", "model path:"+mofile.delete());

                        }

                        File pafile=new File(param_path);
                        if(pafile.exists()){
                            Log.d("remove model data", "param path:"+pafile.delete());

                        }

                        File lbfile=new File(label_path);
                        if(lbfile.exists()){
                            Log.d("remove model data", "label path:"+lbfile.delete());

                        }
                runOnUiThread(new Runnable(){
                    @Override
                    public void run() {
                        // 2.更新UI
                        Toast.makeText(act, "Load model finished, enjoy!", Toast.LENGTH_LONG).show();
                    }
                });
            }
        }.start();

    }

    public String getAssetsCacheFile(Activity activity, String fileName) {

        File cacheFile = new File(activity.getCacheDir(), fileName);
        try {
            //InputStream inputStream = this.getAssets().open(fileName);
            InputStream inputStream=activity.getClass().getClassLoader().getResourceAsStream ("assets/"+fileName);
            try {
                FileOutputStream outputStream = new FileOutputStream(cacheFile);
                try {
                    byte[] buf = new byte[1024];
                    int total=0;
                    int len;
                    while ((len = inputStream.read(buf)) > 0) {
                        outputStream.write(buf, 0, len);
                        outputStream.flush();
                        total+=len;
                    }

                    Log.d("getAssetsCacheFile", fileName+",total bytes:"+total);
                } finally {
                    outputStream.close();
                }
            } finally {
                inputStream.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return cacheFile.getAbsolutePath();
    }

    private void showAvailableBytes(InputStream in) {
        try {
            System.out.println("当前字节输入流中的字节数为:" + in.available());
            Toast.makeText(this,"当前字节输入流中的字节数为:" + in.available(),  Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onBackPressed() {
        DrawerLayout drawer = findViewById(R.id.drawer_layout);
        if (drawer.isDrawerOpen(GravityCompat.START)) {
            drawer.closeDrawer(GravityCompat.START);
        } else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    //==============================================
    // image-fragment
    public  void onFragmentInteraction(String uri){
        Log.d("button", "button clicked,uri:"+uri);
        Toast.makeText(this, "Clicked "+ uri, Toast.LENGTH_LONG).show();
        if(uri=="select_image"){

            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("*/*");//设置类型，我这里是任意类型，任意后缀的可以这样写。
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            try{
                startActivityForResult(Intent.createChooser(intent,"请选择图片") ,FILE_SELECT_CODE);
            }catch (android.content.ActivityNotFoundException ex){
                Toast.makeText(this, "亲，木有文件管理器啊-_-!!", Toast.LENGTH_SHORT).show();
            }
        }else if (uri=="infer_image"){
            if (image_path.isEmpty()||net_env==0){
                Toast.makeText(this, "Invalid image path:"+image_path+"or env:"+net_env, Toast.LENGTH_SHORT).show();
                return ;
            }

           String result=net.inference(net_env,image_path,3);
            if (result.isEmpty()){
                Toast.makeText(this, "Invalid inference result!", Toast.LENGTH_SHORT).show();
                return;
            }
            clsfragments.showClassifyResult(result);
        }

    }

 //  public String getSelectedImage(){
 //       return image_path;
  // }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        // TODO Auto-generated method stub
        if (resultCode != Activity.RESULT_OK) {
            Log.e("image fragment", "onActivityResult() error, resultCode: " + resultCode);
            super.onActivityResult(requestCode, resultCode, data);
            return;
        }
        if (requestCode == FILE_SELECT_CODE) {
            Uri uri = data.getData();
            clsfragments.showImageNow(uri);
            image_path=getRealPath2(uri);

            Toast.makeText(this, "图片路径"+image_path, Toast.LENGTH_SHORT).show();
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private  String getRealPath2( Uri uri) {

        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    GET_PERMISSION);
        return null;
        }

        String filePath = null;
        //final String docId = DocumentsContract.getDocumentId(uri);
        String wholeID = uri.getPath();

        // 使用':'分割
        String id = wholeID.split(":")[1];

        String[] projection = { MediaStore.Images.Media.DATA };
        String selection = MediaStore.Images.Media._ID + "=?";
        String[] selectionArgs = { id };

        Cursor cursor = this.getContentResolver().query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, projection,
                selection, selectionArgs, null);
        int columnIndex = cursor.getColumnIndex(projection[0]);

        if (cursor.moveToFirst()) {
            filePath = cursor.getString(columnIndex);
        }
        cursor.close();
        return filePath;
    }

    @Override
     public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            if (requestCode == GET_PERMISSION) {
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    //
                    Toast.makeText(this, "Permission accept", Toast.LENGTH_SHORT).show();
                } else {
                    // 如果检查到没有被授权则应该到手机setting里面设置授权，如果未显示授权项则检查是否在AndroidManifest.xml中配置了相应权限，否则删除手机上的app重新部署检查；
                    Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
                }
                return;
            }

      }


    @SuppressWarnings("StatementWithEmptyBody")
    @Override
    public boolean onNavigationItemSelected(MenuItem item) {
        // Handle navigation view item clicks here.
        int id = item.getItemId();
        FragmentTransaction ft= fm.beginTransaction();

        if (id == R.id.nav_home) {
            // Handle the camera action
            if (clsfragments==null){
                clsfragments=new ClassifyFragment();
                ft.add(R.id.ftshow,clsfragments);

            }else{
                ft.show(clsfragments);
            }

        } else if (id == R.id.nav_gallery) {

        } else if (id == R.id.nav_slideshow) {

        } else if (id == R.id.nav_tools) {

        }

        DrawerLayout drawer = findViewById(R.id.drawer_layout);
        drawer.closeDrawer(GravityCompat.START);

        //add by clause
        ft.commit();

        return true;
    }

    private void hideFragment(FragmentTransaction ft) {
        if (clsfragments != null) {
            ft.hide(clsfragments);
        }

    }

}
